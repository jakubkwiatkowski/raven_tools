import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

import raven_utils as rv
from core_tools.core import Predict
from core_tools import  ops as K
from raven_utils.models.uitls_ import RangeMask
def interleave(a):
    return tf.reshape(
        tf.concat([i[..., tf.newaxis] for i in a], axis=-1),
        [tf.shape(a[0])[0], -1])


class PredictModel(Model):
    def __init__(self):
        super().__init__()
        self.predict_fn = Predict()
        self.predict_fn_2 = Lambda(lambda x: tf.sigmoid(x) > 0.5)
        self.range_mask = RangeMask()

    # self.predict_fn = partial(tf.argmax, axis=-1)

    def call(self, inputs):
        group_output, output_slot, *properties = rv.decode.output(inputs, tf.split, self.predict_fn, self.predict_fn_2)
        number_loss = K.int64(K.sum(output_slot))
        # predicted properties are rearrange to match target [color_0,size_0, type_0, color_1, size_1, type_1, ...]
        result = tf.concat(
            [group_output[:, None], tf.cast(output_slot, dtype=tf.int64), interleave(properties), number_loss[:, None]],
            axis=-1)

        range_mask = self.range_mask(group_output)
        return [result, range_mask]
        # return [result, range_mask, range_mask, range_mask, range_mask]

# class PredictModel(Model):
#     def __init__(self):
#         super().__init__()
#         self.predict_fn = Lambda(partial(tf.argmax, axis=-1))
#         self.predict_fn_2 = Lambda(lambda x: tf.sigmoid(x) > 0.5)
#         self.range_mask = RangeMask()
#
#     # self.predict_fn = partial(tf.argmax, axis=-1)
#
#     def call(self, inputs):
#         group_output = inputs[rv.OUTPUT_GROUP_SLICE]
#         group_loss = self.predict_fn(group_output)[:, None]
#
#         output_slot = inputs[rv.OUTPUT_SLOT_SLICE]
#         range_mask = self.range_mask(group_loss[:, 0])
#         loss_slot = tf.cast(self.predict_fn_2(output_slot), dtype=tf.int64)
#
#         properties_output = inputs[rv.OUTPUT_PROPERTIES_SLICE]
#         properties = []
#         outputs = tf.split(properties_output, list(rv.ENTITY_PROPERTIES_INDEX.values()), axis=-1)
#         for i, out in enumerate(outputs):
#             shape = (-1, rv.ENTITY_SUM, rv.ENTITY_PROPERTIES_VALUES[i])
#             out_reshaped = tf.reshape(out, shape)
#             properties.append(self.predict_fn(out_reshaped))
#         number_loss = tf.reduce_sum(loss_slot, axis=-1, keepdims=True)
#
#         result = tf.concat([group_loss, loss_slot, interleave(properties), number_loss], axis=-1)
#
#         return [result, range_mask, range_mask, range_mask, range_mask]
