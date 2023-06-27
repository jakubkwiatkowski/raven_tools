import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.keras import Model
import raven_utils as rv


class RangeMask(Model):
    def __init__(self):
        super().__init__()
        ranges = tf.tile(tf.range(rv.entity.INDEX[-1])[None], [rv.group.NO, 1])
        start_index = rv.entity.INDEX[:-1][:, None]
        end_index = rv.entity.INDEX[1:][:, None]
        self.mask = tnp.array((start_index <= ranges) & (ranges < end_index))

    def call(self, inputs):
        return self.mask[inputs]