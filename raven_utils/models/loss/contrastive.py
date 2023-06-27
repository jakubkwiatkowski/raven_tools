import tensorflow as tf
from data_utils import TARGET, OUTPUT, LABELS, PREDICT
from models_utils.models.contrastive_loss import CosineSimilarity
from tensorflow.keras import Model

from models_utils import ops as K, sym_div
from raven_utils.models.loss.class_ import ClassRavenModel
from raven_utils.models.loss.prob_loss import ProbLoss
from raven_utils.models.loss.mask import create_uniform_mask, create_all_mask
from raven_utils.models.uitls_ import RangeMask


# def reverse_index_loss(y_true, y_pred,depth=8):
#     mask = tf.where(tf.one_hot(y_true, depth=depth, on_value=True,off_value=False, dtype=tf.bool), 1, -1)
#     return tf.reduce_mean(mask * y_pred)
#
#
# def contrastive_loss(y_true, y_pred, temperature=0.1):
#     return tf.nn.softmax_cross_entropy_with_logits(y_true, -(y_pred/temperature))


class ContrastiveMetric(Model):
    def __init__(
            self,
            mode=create_all_mask,
            enable_metrics="dist_prob",
            batch_metrics=False,
            **kwargs
    ):
        super().__init__()
        self.range_mask = RangeMask()
        self.mode = mode
        self.class_fn = CosineSimilarity()
        self.batch_metrics = batch_metrics
        self.enable_metrics = enable_metrics
        if self.enable_metrics:
            self.enable_metrics = f"{self.enable_metrics}_" if isinstance(self.enable_metrics, str) else ""
        self.loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    # self.predict_fn = partial(tf.argmax, axis=-1)

    # INDEX, PREDICT, LABELS
    def call(self, inputs):
        metrics = []

        diff = tf.vectorized_map(
            self.class_fn,
            [
                tf.stack(inputs[OUTPUT][:8], axis=1),
                inputs[OUTPUT][-1][:, None]
            ]
        )[..., 0]

        index = tf.argmax(diff, axis=-1)
        if self.loss_fn:
            loss = self.loss_fn(K.int32(inputs['index'][:, 0] - 8), diff)
            self.add_loss(loss)
        acc_batch = (inputs['index'][:, 0] - 8) == index
        acc = K.mean(acc_batch)
        self.add_metric(acc, name=f"{self.enable_metrics}acc")
        metrics.append(acc_batch if self.batch_metrics else acc)

        return {
            **inputs,
            f"{self.enable_metrics}metric": metrics,
            f"{self.enable_metrics}output": index
        }
