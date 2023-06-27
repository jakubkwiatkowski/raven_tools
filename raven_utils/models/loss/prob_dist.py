import tensorflow as tf
from data_utils import TARGET, OUTPUT, LABELS, PREDICT
from tensorflow.keras import Model

from models_utils import ops as K, sym_div
from raven_utils.models.loss.class_ import ClassRavenModel
from raven_utils.models.loss.prob_loss import ProbLoss
from raven_utils.models.loss.mask import create_uniform_mask, create_all_mask
from raven_utils.models.uitls_ import RangeMask

CONTRASTIVE = "contrastive"

REVERSE_INDEX = "reverse_index"


def reverse_index_loss(y_true, y_pred, depth=8):
    mask = tf.where(tf.one_hot(y_true, depth=depth, on_value=True, off_value=False, dtype=tf.bool), 1, -1)
    return tf.reduce_mean(mask * y_pred)


def contrastive_loss(y_true, y_pred, temperature=0.1):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(K.int32(y_true), tf.nn.softmax(-(y_pred / temperature))))


class ProbDistMetric(Model):
    def __init__(
            self,
            mode=create_all_mask,
            enable_metrics="dist_prob",
            batch_metrics=False,
            plw=None,
            loss_type=None,
            loss_sym=False,
            loss_fn=None,
            mask=PREDICT,
    ):
        super().__init__()
        self.range_mask = RangeMask()
        self.mode = mode
        self.class_fn = ProbLoss(enable_metrics=False, mode=mode, loss_type=loss_type, loss_sym=loss_sym, plw=plw,
                                 mask=mask, sparse=False)
        self.batch_metrics = batch_metrics
        self.enable_metrics = enable_metrics
        if self.enable_metrics:
            self.enable_metrics = f"{self.enable_metrics}_" if isinstance(self.enable_metrics, str) else ""
        if loss_fn == REVERSE_INDEX:
            loss_fn = reverse_index_loss
        elif loss_fn == CONTRASTIVE:
            loss_fn = contrastive_loss
        self.loss_fn = loss_fn

    # self.predict_fn = partial(tf.argmax, axis=-1)

    # INDEX, PREDICT, LABELS
    def call(self, inputs):
        metrics = []

        # indexes =[ ]
        # diff = tf.zeros(tf.shape(inputs[OUTPUT])[0])
        diff = tf.zeros((tf.shape(inputs[OUTPUT][0])[0], 8))
        for i in range(8):
            result = self.class_fn([inputs[OUTPUT][i], inputs[OUTPUT][-1]])
            for r in result:
                index = tf.stack((r[0], tf.tile([i], tf.shape(r[0]))), axis=-1)
                diff = tf.tensor_scatter_nd_add(diff, index, r[1])
                # diff = tf.tensor_scatter_nd_add(diff,r[0][:,None],r[1])
        index = tf.argmin(diff, axis=-1)
        if self.loss_fn:
            loss = self.loss_fn(inputs['index'][:, 0] - 8, diff)
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
