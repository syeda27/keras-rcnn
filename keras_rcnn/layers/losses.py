import keras.layers

import keras_rcnn.backend
import tensorflow


class ClassificationLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(ClassificationLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        target, output = inputs

        loss = self.compute_loss(output, target)

        self.add_loss(loss, inputs=inputs)

        return output

    @staticmethod
    def compute_loss(output, target):
        output = keras.backend.reshape(output, [-1, 2])

        condition = keras.backend.not_equal(target, -1)

        indices = keras_rcnn.backend.where(condition)

        output = tensorflow.gather_nd(output, indices)
        target = tensorflow.gather_nd(target, indices)

        loss = keras.backend.mean(keras.backend.sparse_categorical_crossentropy(output, target))

        return loss

    def compute_output_shape(self, input_shape):
        return None, None, None, 18


class RegressionLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(RegressionLoss, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        rpn_bbox_targets, rpn_regression, rpn_labels = inputs

        loss = self.compute_loss(rpn_bbox_targets, rpn_regression, rpn_labels)

        self.add_loss(loss, inputs=inputs)

        return inputs[1]

    @staticmethod
    def compute_loss(rpn_bbox_targets, rpn_regression, rpn_labels):
        # Robust L1 Loss
        rpn_regression = keras.backend.reshape(rpn_regression, [-1, 4])
        rpn_regression = tensorflow.gather_nd(rpn_regression, keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, -1)))
        rpn_bbox_targets = tensorflow.gather_nd(rpn_bbox_targets, keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, -1)))
        rpn_labels = tensorflow.gather_nd(rpn_labels, keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, -1)))

        x = rpn_bbox_targets - rpn_regression

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = keras.backend.cast(keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, 0), keras.backend.ones_like(rpn_labels), keras.backend.zeros_like(rpn_labels)), keras.backend.floatx())

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = tensorflow.matmul(keras.backend.expand_dims(a_x, 0), a_y)
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        loss = 1.0 * (a / b)

        return loss

    def compute_output_shape(self, input_shape):
        return None, None, None, 36
