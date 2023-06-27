from functools import partial

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

import raven_utils as rv
import raven_utils.decode
from data_utils import OUTPUT, PREDICT, TARGET, LOSS
from models_utils import SubClassingModel, DictModel, add_loss, reshape
from models_utils.models.utils import interleave
from raven_utils.config.constant import MASK, LABELS, INDEX
from raven_utils.models.loss.predict import PredictModel
from raven_utils.models.loss.sim import SimilarityRaven
from raven_utils.models.loss.class_ import ClassRavenModel
from raven_utils.models.loss.mask import create_all_mask, create_uniform_mask
from raven_utils.models.uitls_ import RangeMask


class BaselineClassificationLossModel(Model):
    def __init__(self, mode=create_all_mask, number_loss=False, slot_loss=True, group_loss=True):
        super().__init__()
        self.predict_fn = SubClassingModel([lambda x: x[0], PredictModel()])
        self.loss_fn = ClassRavenModel(mode=mode, number_loss=number_loss, slot_loss=slot_loss,
                                       group_loss=group_loss)
        self.metric_fn = SimilarityRaven(mode=mode)

    def call(self, inputs):
        losses = []
        output = inputs[1]
        losses.append(self.loss_fn([inputs[0][0], output]))
        losses.append(self.metric_fn([inputs[0][2], inputs[3][0], inputs[0][1][:, 8:]]))
        return losses


class SingleRavenLoss(Model):
    def __init__(self, mode=create_all_mask, number_loss=False, slot_loss=True, group_loss=True, lw=(1.0, 0.1),
                 classification=False, trans=True, anneal=False):
        super().__init__()
        if anneal:
            self.weight_scheduler
        self.classification = classification
        self.trans = trans
        self.predict_fn = DictModel(PredictModel(), in_=OUTPUT, out=[PREDICT, MASK], name="pred")
        self.loss_fn = add_loss(ClassRavenModel(mode=mode, number_loss=number_loss, slot_loss=slot_loss,
                                                group_loss=group_loss), lw=lw[0], name="add_loss")
        self.metric_fn = SimilarityRaven(mode=mode)

    def call(self, inputs):
        losses = []
        output = inputs[OUTPUT]
        target = inputs[TARGET]
        labels = inputs[LABELS]

        losses.append(self.loss_fn([target, output]))
        losses.append(self.metric_fn([inputs[INDEX], inputs[PREDICT], labels]))
        return {**inputs, LOSS: losses}


class FullMask(Model):
    def __init__(self, mode=create_uniform_mask):
        super().__init__()
        self.range_mask = RangeMask()
        self.mode = mode

    def call(self, inputs):
        target_group, target_slot, _ = raven_utils.decode.decode_target(inputs)
        full_properties_musks = self.mode(inputs)
        range_mask = self.range_mask(target_group)

        number_mask = range_mask & full_properties_musks[0]

        slot_mask = range_mask & full_properties_musks[1]
        properties_mask = []
        for property_mask in full_properties_musks[2:]:
            properties_mask.append(tf.cast(target_slot, "bool") & property_mask)
        return [slot_mask, properties_mask, number_mask]


class PredictModelMasked(Model):
    def __init__(self):
        super().__init__()
        self.predict_fn = Lambda(partial(tf.argmax, axis=-1))
        self.loss_fn_2 = Lambda(lambda x: tf.sigmoid(x) > 0.5)
        self.range_mask = RangeMask()

    # self.predict_fn = partial(tf.argmax, axis=-1)

    def call(self, inputs):
        group_output = inputs[:, -rv.GROUPS_NO:]
        group_loss = self.predict_fn(group_output)[:, None]

        output_slot = inputs[:, :rv.ENTITY_SUM]
        range_mask = self.range_mask(group_loss[:, 0])
        loss_slot = tf.cast(self.predict_fn_2(output_slot * range_mask), dtype=tf.int64)

        properties_output = inputs[:, rv.ENTITY_SUM:-rv.GROUPS_NO]

        properties = []
        outputs = tf.split(properties_output, list(rv.ENTITY_PROPERTIES_INDEX.values()), axis=-1)
        for i, out in enumerate(outputs):
            shape = (-1, rv.ENTITY_SUM, rv.ENTITY_PROPERTIES_VALUES[i])
            out_reshaped = tf.reshape(out, shape)
            out_masked = out_reshaped * loss_slot[..., None]
            properties.append(self.predict_fn(out_masked))
            # out_masked[0].numpy()
        number_loss = tf.reduce_sum(loss_slot, axis=-1, keepdims=True)

        result = tf.concat([group_loss, loss_slot, interleave(properties), number_loss], axis=-1)

        return result


def final_predict_mask(x, mask):
    r = reshape(x[0][:, rv.INDEX[0]:-1], [-1, 3])
    return tf.ragged.boolean_mask(r, mask)


def final_predict(x, mode=False):
    m = x[1] if mode else tf.cast(x[0][:, 1:rv.INDEX[0]], tf.bool)
    return final_predict_mask(x[0], m)


def final_predict_2(x):
    ones = tf.cast(tf.ones(tf.shape(x[0])[0]), tf.bool)[:, None]
    mask = tf.concat([ones, tf.tile(x[1], [1, 4]), ones], axis=-1)
    return tf.ragged.boolean_mask(x[0], mask)
