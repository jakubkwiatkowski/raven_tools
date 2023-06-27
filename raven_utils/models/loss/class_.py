import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, mse
from tensorflow.keras.metrics import SparseCategoricalAccuracy, BinaryAccuracy

import raven_utils as rv
import raven_utils.decode
from raven_utils.config.constant import ACC, GROUP, SLOT, NUMBER, PROPERTIES
from raven_utils.models.loss.predict import PredictModel
from raven_utils.models.loss.mask import create_uniform_mask, create_uniform_num_pos_mask, \
    create_uniform_num_pos_arth_mask
from raven_utils.models.uitls_ import RangeMask


class ClassRavenModel(Model):
    def __init__(
            self,
            mode=create_uniform_num_pos_arth_mask,
            plw=None,
            number_loss=False,
            slot_loss=True,
            group_loss=True,
            enable_metrics=True,
            lw=1.0,
            return_prop_mask=False,

    ):
        super().__init__()
        self.number_loss = number_loss
        self.group_loss = group_loss
        self.enable_metrics = enable_metrics
        self.slot_loss = slot_loss
        self.predict_fn = PredictModel()
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        if self.slot_loss:
            self.loss_fn_2 = tf.nn.sigmoid_cross_entropy_with_logits
        if self.enable_metrics:
            self.enable_metrics = f"{self.enable_metrics}_" if isinstance(self.enable_metrics, str) else ""
            self.metric_fn = [
                SparseCategoricalAccuracy(name=f"{self.enable_metrics}{ACC}_{property_}") for property_ in
                rv.properties.NAMES]
            if self.group_loss:
                self.metric_fn_group = SparseCategoricalAccuracy(name=f"{self.enable_metrics}{ACC}_{GROUP}")
            if self.slot_loss:
                self.metric_fn_2 = BinaryAccuracy(name=f"{self.enable_metrics}{ACC}_{SLOT}")
        self.range_mask = RangeMask()
        self.mode = mode
        self.lw = lw
        if not plw:
            plw = [1., 95.37352927, 2.83426987, 0.85212836, 1.096005, 1.21943385]
        elif isinstance(plw, int) or isinstance(plw, float):
            plw = [1., plw, 2.83426987, 0.85212836, 1.096005, 1.21943385]
            # plw = [plw] * 6
        self.plw = plw
        self.return_prop_mask = return_prop_mask

    # self.predict_fn = partial(tf.argmax, axis=-1)

    def call(self, inputs):
        losses = []
        metrics = {}
        target = inputs[0]
        output = inputs[1]

        target_group, target_slot, target_all = raven_utils.decode.decode_target(target)

        group_output, output_slot, outputs = raven_utils.decode.output_divide(output, split_fn=tf.split)

        # group
        if self.group_loss:
            group_loss = self.lw * self.plw[0] * self.loss_fn(target_group, group_output)
            losses.append(group_loss)

            if isinstance(self.enable_metrics, str):
                group_metric = self.metric_fn_group(target_group, group_output)
                # metrics[GROUP] = group_metric
                self.add_metric(group_metric)
                self.add_metric(tf.reduce_sum(group_metric), f"{self.enable_metrics}{ACC}")

        # setting uniformity mask
        # create mask for each attribute that loss is calculated (list of tensors) [number, slot,color, size, type]
        full_properties_musks = self.mode(target)

        range_mask = self.range_mask(target_group)

        if self.slot_loss:
            # number
            number_mask = range_mask & full_properties_musks[0]
            number_mask = tf.cast(number_mask, tf.float32)
            target_number = tf.reduce_sum(
                tf.cast(target_slot, "float32") * number_mask, axis=-1)
            output_number = tf.reduce_sum(
                tf.cast(tf.sigmoid(output_slot) >= 0.5, "float32") * number_mask, axis=-1)

            # output_number = tf.reduce_sum(tf.sigmoid(output_slot) * number_mask, axis=-1)
            if self.number_loss:
                scale = 1 / 9
                if self.number_loss == 2:
                    output_number_2 = tf.reduce_sum(tf.sigmoid(output_slot) * number_mask, axis=-1)
                else:
                    output_number_2 = output_number
                number_loss = self.lw * self.plw[1] * mse(tf.stop_gradient(target_number) * scale,
                                                          output_number_2 * scale)
                losses.append(number_loss)

            # metrics[NUMBER] = number_acc

            if isinstance(self.enable_metrics, str):
                number_acc = tf.reduce_mean(
                    tf.cast(tf.cast(target_number, "int8") == tf.cast(output_number, "int8"), "float32"))
                self.add_metric(tf.reduce_sum(number_acc), f"{self.enable_metrics}{ACC}_{NUMBER}")
                # self.add_metric(tf.reduce_sum(number_acc), f"{self.enable_metrics}{ACC}")
                # self.add_metric(tf.reduce_sum(number_acc), f"{self.enable_metrics}{ACC}_NO_{GROUP}")

            # position/slot
            # There difference between range mask and target_slot it that range mask shows all possible slots for group/arrangament while the target_slot shows only slots that are filled in certain task.
            # So, in slot loss we interested if model predict filled slot. target_slot/range_slot
            # In properties we only interested in filled slots/existing objects, so we use target_slot.
            slot_mask = range_mask & full_properties_musks[1]
            # tf.boolean_mask(target_slot,slot_mask)

            if tf.reduce_any(slot_mask):
                # if tf.reduce_mean(tf.cast(slot_mask, dtype=tf.int32)) > 0:
                target_slot_masked = tf.boolean_mask(target_slot, slot_mask)[:, None]
                output_slot_masked = tf.boolean_mask(output_slot, slot_mask)[:, None]
                loss_slot = self.lw * self.plw[2] * tf.reduce_mean(
                    self.loss_fn_2(tf.cast(target_slot_masked, "float32"), output_slot_masked))
                if isinstance(self.enable_metrics, str):
                    acc_slot = self.metric_fn_2(target_slot_masked, tf.sigmoid(output_slot_masked))
                    self.add_metric(acc_slot)
                    self.add_metric(tf.reduce_sum(acc_slot), f"{self.enable_metrics}{ACC}")
                    self.add_metric(tf.reduce_sum(acc_slot), f"{self.enable_metrics}{ACC}_NO_{GROUP}")
            else:
                loss_slot = 0.0
                acc_slot = -1.0

            losses.append(loss_slot)
            # metrics[SLOT] = acc_slot
        # if loss_slot != 0:

        # if tf.reduce_any(slot_mask):

        # self.add_metric(acc_slot, f"{self.enable_metrics}{ACC}_{NUMBER}")
        # self.add_metric(acc_slot, f"{self.enable_metrics}{ACC}")
        # self.add_metric(acc_slot, f"{self.enable_metrics}{ACC}_NO_{GROUP}")

        # properties
        for i, out in enumerate(outputs):
            shape = (-1, rv.entity.SUM, rv.properties.RAW_SIZE[i])
            out_reshaped = tf.reshape(out, shape)
            properties_mask = tf.cast(target_slot, "bool") & full_properties_musks[i + 2]

            if tf.reduce_any(properties_mask):
                out_masked = tf.boolean_mask(out_reshaped, properties_mask)
                out_target = tf.boolean_mask(target_all[i], properties_mask)
                loss = self.lw * self.plw[3 + i] * self.loss_fn(out_target, out_masked)
                if isinstance(self.enable_metrics, str):
                    metric = self.metric_fn[i](out_target, out_masked)
                    self.add_metric(metric)
                    # self.add_metric(metric, f"{self.enable_metrics}{ACC}")
                    self.add_metric(tf.reduce_sum(metric), f"{self.enable_metrics}{ACC}")
                    self.add_metric(tf.reduce_sum(metric), f"{self.enable_metrics}{ACC}_{PROPERTIES}")
                    self.add_metric(tf.reduce_sum(metric), f"{self.enable_metrics}{ACC}_NO_{GROUP}")
            else:
                loss = 0.0
                metric = -1.0

            losses.append(loss)
        if self.return_prop_mask:
            return losses, full_properties_musks
        return losses
