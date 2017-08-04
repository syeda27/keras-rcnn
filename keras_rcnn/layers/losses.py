import keras.layers

import keras_rcnn.backend


class ClassificationLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(ClassificationLoss, self).__init__(**kwargs)

    def _loss(self, rpn_labels, rpn_classification):
        rpn_classification = keras.backend.reshape(rpn_classification, [-1, 2])
        rpn_classification = keras.backend.gather(rpn_classification, keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, -1)))
        rpn_labels         = keras.backend.gather(rpn_labels, keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, -1)))
        loss               = keras.backend.mean(keras.backend.sparse_categorical_crossentropy(rpn_labels, rpn_classification))

        return loss

    def call(self, inputs):
        rpn_labels, rpn_classification = inputs

        loss = self._loss(rpn_labels, rpn_classification)

        self.add_loss(loss, inputs=inputs)

        return loss



class RegressionLoss(keras.layers.Layer):
    def __init__(self, anchors, **kwargs):
        self.anchors = anchors

        super(RegressionLoss, self).__init__(**kwargs)

    def _loss(self, rpn_bbox_targets, rpn_regression, rpn_labels):
        # Robust L1 Loss
        rpn_regression = keras.backend.reshape(rpn_regression, [-1, 4])
        rpn_regression = keras.backend.gather(rpn_regression, keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, -1)))
        rpn_labels     = keras.backend.gather(rpn_labels, keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, -1)))

        x = rpn_bbox_targets - rpn_regression

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = keras.backend.cast(keras_rcnn.backend.where(keras.backend.not_equal(rpn_labels, 0))[1][0], keras.backend.floatx())

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = a_x * a_y
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        loss = 1.0 * (a / b)

        return loss

    def call(self, inputs):
        y_true, y_pred, labels = inputs

        loss = self._loss(y_true, y_pred, labels)

        self.add_loss(loss, inputs=inputs)

        return loss