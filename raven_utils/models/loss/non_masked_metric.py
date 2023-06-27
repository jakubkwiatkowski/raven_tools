import tensorflow as tf
from tensorflow.keras import Model

import raven_utils as rv
from experiment_utils.constants import METRIC
from models_utils import BatchModel, ops as K
from raven_utils.config.constant import ACC_SAME
from raven_utils.constant import HAMF, HAM, HAMS, T_ACC, AVG_PROP
from raven_utils.models.loss.predict import PredictModel
from raven_utils.models.loss.mask import create_uniform_mask
from raven_utils.models.uitls_ import RangeMask


class NonMaskedMetricRaven(Model):
    def __init__(self, mode=create_uniform_mask, enable_metrics="c"):
        super().__init__()
        self.range_mask = RangeMask()
        self.mode = mode
        self.predict_fn = BatchModel(PredictModel())
        self.enable_metrics = enable_metrics
        if self.enable_metrics:
            self.enable_metrics = f"{self.enable_metrics}_" if isinstance(self.enable_metrics, str) else ""

    # self.predict_fn = partial(tf.argmax, axis=-1)

    # INDEX, PREDICT, LABELS
    def call(self, inputs):
        metrics = []

        target = inputs['target'][:, :8]
        predict = self.predict_fn(inputs['output'][:, :8])[0]
        shape = tf.shape(predict)

        target_group = target[:, :, 0]
        target_slot = target[..., 1:rv.target.INDEX[0]]

        # comp_slice = np.
        target_comp = target[:, :, 1:rv.target.END_INDEX]
        predict_comp = predict[:, :, 1:rv.target.END_INDEX]

        range_mask = self.range_mask(target_group)
        full_range_mask = K.cat([range_mask, tf.repeat(target_slot, 3, axis=-1)], axis=-1)

        final_mask = full_range_mask

        target_masked = target_comp * final_mask
        predict_masked = predict_comp * final_mask

        same = target_masked == predict_masked
        diff = ~same
        acc_same = tf.reduce_mean(K.float32(K.all(same)))
        self.add_metric(acc_same, f"{self.enable_metrics}{ACC_SAME}")
        metrics.append(acc_same)

        hamf = tf.reduce_mean(K.sum(predict[:, :, :101] != target[:, :, :101]))
        self.add_metric(hamf, f"{self.enable_metrics}{HAMF}")
        metrics.append(hamf)

        ham_sum = K.sum(diff) + (target_group != predict[:, :, 0])
        total_predictions = tf.reduce_sum(K.sum(final_mask) + 1)
        accuracy = 1 - (tf.reduce_sum(ham_sum) / total_predictions)
        self.add_metric(accuracy, f"{self.enable_metrics}{T_ACC}")
        metrics.append(accuracy)

        ham = tf.reduce_mean(ham_sum)
        self.add_metric(ham, f"{self.enable_metrics}{HAM}")
        metrics.append(ham)

        # hams = tf.reduce_mean(ham_sum / K.array(list(rv.properties.COUNT_ALL.values()))[target_group])
        hams = tf.reduce_mean(ham_sum / K.sum(final_mask))
        self.add_metric(hams, f"{self.enable_metrics}{HAMS}")
        metrics.append(hams)
        self.add_metric(1 - hams, f"{self.enable_metrics}{AVG_PROP}")
        metrics.append(1 - hams)

        return {**inputs, f"{self.enable_metrics}{METRIC}": metrics}
