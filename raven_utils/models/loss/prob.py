import tensorflow as tf
from data_utils import TARGET, OUTPUT, LABELS
from tensorflow.keras import Model

from models_utils import ops as K
from raven_utils.models.loss.class_ import ClassRavenModel
from raven_utils.models.loss.prob_loss import ProbLoss
from raven_utils.models.loss.mask import create_uniform_mask, create_all_mask, create_uniform_num_pos_arth_mask
from raven_utils.models.uitls_ import RangeMask


class ProbMetric(Model):
    def __init__(self, mode=create_uniform_num_pos_arth_mask, enable_metrics="prob", batch_metrics=False,
                 loss_type=None,
                 mask=TARGET):
        super().__init__()
        self.range_mask = RangeMask()
        self.mode = mode
        self.class_fn = ProbLoss(enable_metrics=False, loss_type=loss_type, mask=mask, sparse=True)
        self.batch_metrics = batch_metrics
        self.enable_metrics = enable_metrics
        if self.enable_metrics:
            self.enable_metrics = f"{self.enable_metrics}_" if isinstance(self.enable_metrics, str) else ""

    # self.predict_fn = partial(tf.argmax, axis=-1)

    # INDEX, PREDICT, LABELS
    def call(self, inputs):
        metrics = []
        # print(f"------------------------------")
        # print(f"{self.enable_metrics}")

        # indexes =[ ]
        # diff = tf.zeros(tf.shape(inputs[OUTPUT])[0])
        diff = tf.zeros((tf.shape(inputs[OUTPUT])[0], 8))
        for i in range(8):
            result = self.class_fn([inputs[LABELS][:, i + 8], inputs[OUTPUT][:, -1]])
            for r in result:
                index = tf.stack((r[0], tf.tile([i], tf.shape(r[0]))), axis=-1)
                diff = tf.tensor_scatter_nd_add(diff, index, r[1])
                # diff = tf.tensor_scatter_nd_add(diff,r[0][:,None],r[1])
        index = tf.argmin(diff, axis=-1)
        acc_batch = (inputs['index'][:, 0] - 8) == index
        acc = K.mean(acc_batch)
        self.add_metric(acc, name=f"{self.enable_metrics}acc")
        metrics.append(acc_batch if self.batch_metrics else acc)

        return {
            **inputs,
            f"{self.enable_metrics}metric": metrics,
            f"{self.enable_metrics}output": index
        }

