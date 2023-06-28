import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from core_tools.core import IndexReshape, METRICS, TakeDict, INDEX, PREDICT, OUTPUT
from raven_utils.constant import ANSWER, DIRECT, FULL_JOIN, SEPARATE

from raven_utils.models.loss import SimilarityRaven, create_all_mask
CLASS_IMAGE = "class_image"

class ChoiceMaker1(Model):
    def __init__(self, model, model2, tail, select_out=1):
        super().__init__()
        self.model = model
        self.model2 = model2
        self.select_out = select_out
        # self.mask_fn = ImageMask(last=take_by_index)
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.acc_fn = SparseCategoricalAccuracy()
        self.dense = Dense(256, "relu")
        self.flatten = Flatten()
        self.tail = tail
        self.index_reshape = IndexReshape((0, (1, 3), None))

    def call(self, inputs):
        images = inputs['inputs']
        base = self.model(images)
        # x = tf.concat([images[:, 8:], images[:, i + 8:i + 9]], axis=1)
        x = images[:, 8:]
        r = self.dense(self.index_reshape(self.model2(x))[:, :8])

        if self.select_out == 1:
            x = self.tail(self.flatten(tf.concat([base[:, None], r], axis=1)))
        else:
            x = self.flatten(self.tail(tf.concat([tf.tile(base[:, None], (1, 8, 1)), r], axis=-1)))

        target = inputs['index'][:, 0] - 8
        self.add_loss(self.loss_fn(target, x))
        self.add_metric(self.acc_fn(target, x))
        return x


class ChoiceMaker2(Model):
    def __init__(self, model, model2, tail, select_out=1):
        super().__init__()
        self.model = model
        self.model2 = model2
        self.select_out = select_out
        # self.mask_fn = ImageMask(last=take_by_index)
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.acc_fn = SparseCategoricalAccuracy()
        self.flatten = Flatten()
        self.tail = tail

    def call(self, inputs):
        base = self.model(TakeDict(inputs)[:, :8])['output']
        r = self.model2(TakeDict(inputs)[:, 8:])['output']

        if self.select_out == 1:
            x = self.tail(self.flatten(tf.concat([base[:, None], r], axis=1)))
        else:
            x = self.flatten(self.tail(tf.concat([tf.tile(base[:, None], (1, 8, 1)), r], axis=-1)))

        target = inputs['index'][:, 0] - 8
        self.add_loss(self.loss_fn(target, x))
        self.add_metric(self.acc_fn(target, x))
        return x


class ChoiceMaker(Model):
    def __init__(self, model, model2, predictor, select_out=1, ):
        super().__init__()
        self.model = model
        self.model2 = model2
        self.select_out = select_out
        # self.mask_fn = ImageMask(last=take_by_index)
        self.predictor = predictor

    def call(self, inputs):
        images = inputs['inputs']
        result = []
        base = self.model(inputs)["output"]
        class_image = []
        for i in range(8):
            index = tf.ones_like(inputs['index']) * (i + 8)
            st_input = {**inputs, 'index': index}
            r = self.model2(st_input)['output']
            if self.select_out == SEPARATE:
                result.append(self.predictor(tf.concat([base, r], axis=-1)))
            else:
                result.append(r)
            class_image.append(r)
        if self.select_out == FULL_JOIN:
            x = self.predictor(tf.concat([base] + result, axis=-1))
        elif self.select_out == DIRECT:
            result.append(base)
            x = result
        else:
            x = tf.concat(result, axis=-1)
        # target = inputs['index'][:, 0] - 8
        # self.add_loss(self.loss_fn(target, x))
        # self.add_metric(self.acc_fn(target, x))
        return {
            **inputs,
            OUTPUT: x,
            CLASS_IMAGE: class_image + [base],
        }


class DCPMetric(Model):
    def __init__(self, mask=ANSWER, batch_metrics=False):
        super().__init__()
        self.metric_fn = SimilarityRaven(mode=create_all_mask,batch_metrics=batch_metrics, mask=mask, mask_slot=True)

    def call(self, inputs):
        labels = tf.tile(tf.transpose(inputs[PREDICT][:-1], (1, 0, 2)), (1, 2, 1))
        x = self.metric_fn([inputs[INDEX], inputs[PREDICT][-1], labels])
        return {**inputs, METRICS: x}
