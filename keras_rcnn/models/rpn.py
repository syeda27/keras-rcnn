# -*- coding: utf-8 -*-

import keras

import keras_rcnn.layers


"""
Faster R-CNN RPN model by S Ren et, al. (2015).

:param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
:param features: (Convolutional) features (e.g. extracted by `keras_resnet.models.ResNet50`)
:param rois: integer, number of regions of interest per image

:return model: a functional model API for R-CNN.
"""

def rpn(inputs, features, rois):
    image, im_info, gt_boxes = inputs
    num_anchors = 9 # TODO: Parametrize this

    # Compute RPN features
    rpn_conv = keras.layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu", name="rpn_conv")(features)

    # Compute classification and regression outputs
    rpn_classification = keras.layers.Conv2D(num_anchors * 2, kernel_size=(1, 1), activation="sigmoid", name="rpn_cls_prob")(rpn_conv)
    rpn_regression     = keras.layers.Conv2D(num_anchors * 4, kernel_size=(1, 1), name="rpn_bbox_pred")(rpn_conv)

    # Compute proposals and RPN targets
    proposals                        = keras_rcnn.layers.ObjectProposal(maximum_proposals=rois, name="proposals")([im_info, rpn_regression, rpn_classification])
    rpn_labels, rpn_bbox_reg_targets = keras_rcnn.layers.AnchorTarget(name="anchor_target")([rpn_classification, gt_boxes, image])

    # Compute RPN losses
    rpn_classification_loss = keras_rcnn.layers.ClassificationLoss(anchors=num_anchors, name="rpn_classification_loss")([rpn_classification, rpn_labels])
    rpn_regression_loss     = keras_rcnn.layers.RegressionLoss(anchors=num_anchors, name="rpn_bbox_loss")([rpn_regression, rpn_bbox_reg_targets, rpn_labels])

    return proposals, rpn_classification_loss, rpn_regression_loss
