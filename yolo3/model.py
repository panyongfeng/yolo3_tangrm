"""YOLO_v3 Model Defined in Keras."""

from yolo3.enum import BOX_LOSS
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from yolo3.utils import compose
from yolo3.override import mobilenet_v2
from yolo3.darknet import DarknetConv2D_BN_Leaky, DarknetConv2D, darknet_body
from yolo3.efficientnet import EfficientNetB4, MBConvBlock, get_model_params, BlockArgs


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def darknet_yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    if not hasattr(inputs, '_keras_history'):
        inputs = tf.keras.layers.Input(tensor=inputs)
    darknet = darknet_body(inputs, include_top=False)
    x, y1 = make_last_layers(darknet.output, 512,
                             num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)),
                tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)),
                tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    y1 = tf.keras.layers.Reshape(
        (tf.shape(y1)[1], tf.shape(y1)[2], num_anchors, num_classes + 5),
        name='y1')(y1)
    y2 = tf.keras.layers.Reshape(
        (tf.shape(y2)[1], tf.shape(y2)[2], num_anchors, num_classes + 5),
        name='y2')(y2)
    y3 = tf.keras.layers.Reshape(
        (tf.shape(y3)[1], tf.shape(y3)[2], num_anchors, num_classes + 5),
        name='y3')(y3)
    return tf.keras.models.Model(inputs, [y1, y2, y3])


def MobilenetSeparableConv2D(filters,
                             kernel_size,
                             strides=(1, 1),
                             padding='valid',
                             use_bias=True):
    return compose(
        tf.keras.layers.DepthwiseConv2D(kernel_size,
                                        padding=padding,
                                        use_bias=use_bias,
                                        strides=strides),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.),
        tf.keras.layers.Conv2D(filters,
                               1,
                               padding='same',
                               use_bias=use_bias,
                               strides=1), tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(6.))


def make_last_layers_mobilenet(x, id, num_filters, out_filters):
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 1) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 1) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 1) + '_relu6'),
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_' + str(id + 2) + '_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9,
                                           name='block_' + str(id + 2) + '_BN'),
        tf.keras.layers.ReLU(6., name='block_' + str(id + 2) + '_relu6'))(x)
    y = compose(
        MobilenetSeparableConv2D(2 * num_filters,
                                 kernel_size=(3, 3),
                                 use_bias=False,
                                 padding='same'),
        tf.keras.layers.Conv2D(out_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobilenetConv2D(kernel, alpha, filters):
    last_block_filters = _make_divisible(filters * alpha, 8)
    return compose(
        tf.keras.layers.Conv2D(last_block_filters,
                               kernel,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(), tf.keras.layers.ReLU(6.))


def mobilenetv2_yolo_body(inputs, num_classes, alpha=1.0):
    mobilenetv2 = mobilenet_v2(default_batchnorm_momentum=0.9,
                               alpha=alpha,
                               input_tensor=inputs,
                               include_top=False,
                               weights='imagenet')

    x = mobilenetv2.get_layer('block_12_project_BN').output
    x, y1 = make_last_layers_mobilenet(x, 21, 256, (num_classes + 3))
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(momentum=0.9, name='block_24_BN'),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    x = tf.keras.layers.Concatenate()([
        x,
        MobilenetConv2D((1, 1), alpha,
                        128)(mobilenetv2.get_layer('block_5_project_BN').output)
    ])
    x, y2 = make_last_layers_mobilenet(x, 25, 128, (num_classes + 3))

    y1 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1], tf.shape(y)[2], num_classes + 3 ]), name='y1')(y1)

    y2 = tf.keras.layers.Lambda(lambda y: tf.reshape(y, [
        -1, tf.shape(y)[1], tf.shape(y)[2], num_classes + 3 ]), name='y2')(y2)


    return tf.keras.models.Model(inputs, [y1, y2])


def make_last_layers_efficientnet(x, block_args, global_params):
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    num_filters = block_args.input_filters * block_args.expand_ratio
    x = compose(
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum,
            fused=False),
        tf.keras.layers.ReLU(6.),
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum,
            fused=False),
        tf.keras.layers.ReLU(6.),
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(num_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False),
        tf.keras.layers.BatchNormalization(
            axis=channel_axis,
            epsilon=global_params.batch_norm_epsilon,
            momentum=global_params.batch_norm_momentum,
            fused=False),
        tf.keras.layers.ReLU(6.))(x)

    y = compose(
        MBConvBlock(block_args,
                    global_params,
                    drop_connect_rate=global_params.drop_connect_rate),
        tf.keras.layers.Conv2D(block_args.output_filters,
                               kernel_size=1,
                               padding='same',
                               use_bias=False))(x)
    return x, y


def efficientnet_yolo_body(inputs, model_name, num_anchors, **kwargs):
    _, global_params, input_shape = get_model_params(model_name, kwargs)
    num_classes = global_params.num_classes
    if global_params.data_format == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1
    efficientnet = EfficientNetB4(include_top=False,
                                  weights='imagenet',
                                  input_shape=(input_shape, input_shape, 3),
                                  input_tensor=inputs)
    print("===================================================")
    print(inputs)
    print(efficientnet.input)
    print("===================================================")
    block_args = BlockArgs(kernel_size=3,
                           num_repeat=1,
                           input_filters=512,
                           output_filters=num_anchors * (num_classes + 5),
                           expand_ratio=1,
                           id_skip=True,
                           se_ratio=0.25,
                           strides=[1, 1])
    #x, y1 = make_last_layers_efficientnet(efficientnet.output, block_args,
    #                                      global_params)
    #x = compose(
    #    tf.keras.layers.Conv2D(256,
    #                           kernel_size=1,
    #                           padding='same',
    #                           use_bias=False,
    #                           name='block_20_conv'),
    #    tf.keras.layers.BatchNormalization(axis=channel_axis,
    #                                       momentum=0.9,
    #                                       name='block_20_BN'),
    #    tf.keras.layers.ReLU(6., name='block_20_relu6'),
    #    tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=256)
    #x = tf.keras.layers.Concatenate()(
    #    [x, efficientnet.get_layer('swish_65').output])

    x = efficientnet.get_layer('swish_65').output
    print("====================x===============================")
    print(x)
    print("===================================================")
    x, y2 = make_last_layers_efficientnet(x, block_args, global_params)
    print("=====================x=y=============================")
    print(x)
    print(y2)
    print("===================================================")
    x = compose(
        tf.keras.layers.Conv2D(128,
                               kernel_size=1,
                               padding='same',
                               use_bias=False,
                               name='block_24_conv'),
        tf.keras.layers.BatchNormalization(axis=channel_axis,
                                           momentum=0.9,
                                           name='block_24_BN',
                                           fused=False),
        tf.keras.layers.ReLU(6., name='block_24_relu6'),
        tf.keras.layers.UpSampling2D(2))(x)
    block_args = block_args._replace(input_filters=128)
    x = tf.keras.layers.Concatenate()(
        [x, efficientnet.get_layer('swish_29').output])
    x, y3 = make_last_layers_efficientnet(x, block_args, global_params)
    print("=====================x=y=============================")
    print(x)
    print(y3)
    print("===================================================")
    #y1 = tf.keras.layers.Reshape(
    #    (y1.shape[1], y1.shape[2], num_anchors, num_classes + 5), name='y1')(y1)
    y2 = tf.keras.layers.Reshape(
        (y2.shape[1], y2.shape[2], num_anchors, num_classes + 5), name='y2')(y2)
    y3 = tf.keras.layers.Reshape(
        (y3.shape[1], y3.shape[2], num_anchors, num_classes + 5), name='y3')(y3)
    #return tf.keras.models.Model(efficientnet.input, [y2, y3])
    return tf.keras.models.Model(inputs, [y2, y3])


def yolo_head(feats: tf.Tensor,
              input_shape: tf.Tensor,
              calc_loss: bool = False
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Convert final layer features to bounding box parameters."""
    # Reshape to batch, height, width, num_anchors, box_params.
    grid_shape = tf.shape(feats)[1:3]
    grid_y = tf.tile(tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1]),
                     [1, grid_shape[1], 1])
    grid_x = tf.tile(tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1]),
                     [grid_shape[0], 1, 1])
    grid = tf.concat([grid_x, grid_y], -1)
    grid = tf.cast(grid, feats.dtype)

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(
        grid_shape[::-1], feats.dtype)
    
    box_confidence = tf.sigmoid(feats[..., 2:3])
    if calc_loss == True:
        return grid, box_xy, box_confidence
    box_class_probs = tf.sigmoid(feats[..., 3:])
    return box_xy, box_confidence, box_class_probs




def yolo_boxes_and_scores(feats: tf.Tensor,
                          num_classes: int, input_shape: Tuple[int, int],
                          image_shape) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Process Conv layer output'''
    box_xy, box_confidence, box_class_probs = yolo_head(
        feats, input_shape)
    
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores, [-1, num_classes])
    return box_xy, box_scores


def yolo_eval(yolo_outputs: List[tf.Tensor],
              num_classes: int,
              image_shape,
              max_boxes: int = 20,
              score_threshold: float = .6,
              iou_threshold: float = .5
             ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
    """Evaluate YOLO model on given input and return filtered boxes."""

    num_layers = len(yolo_outputs)

    input_shape = tf.shape(yolo_outputs[0])[1:3] * 16
    grid_size = [26, 52]
    boxes_xy = []
    box_score = []
    for l in range(num_layers):
        _boxes_xy, _box_score = yolo_boxes_and_scores(yolo_outputs[l],
                                                    num_classes, input_shape,
                                                    image_shape)
        #boxes_xy.append(_boxes_xy)
        #box_score.append(_box_score)
        #box_class_probs.append(_box_class_probs)
        _box_score = tf.reshape(_box_score, [-1, num_classes])
        _boxes_xy = tf.reshape(_boxes_xy, [-1, 2])
        #for i, c in enumerate(_box_score):
        #  idx = np.argmax(c)
        #  if c[idx] > score_threshold:
        #    boxes_xy.append( _boxes_xy[i] )
        #    boxes_score.append(c[idx])
        #    boxes_classes.append(idx)
        boxes_xy.append(_box_score)
        box_score.append(_box_score)

    boxes_xy = tf.concat(boxes_xy, axis=0)
    box_score = tf.concat(box_score, axis=0)



    
    #TODO
    return boxes_xy, box_score, box_score


class YoloEval(tf.keras.layers.Layer):

    def __init__(self,
                 num_classes,
                 image_shape,
                 max_boxes=20,
                 score_threshold=.6,
                 iou_threshold=.5,
                 **kwargs):
        super(YoloEval, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.image_shape = image_shape
        self.max_boxes = max_boxes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def call(self, yolo_outputs):
        return yolo_eval(yolo_outputs, self.num_classes,
                         self.image_shape, self.max_boxes, self.score_threshold,
                         self.iou_threshold)

    def get_config(self):
        config = super(YoloEval, self).get_config()
        config['num_classes'] = self.num_classes
        config['image_shape'] = self.image_shape
        config['max_boxes'] = self.max_boxes
        config['score_threshold'] = self.score_threshold
        config['iou_threshold'] = self.iou_threshold

        return config


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_giou(b1, b2):
    # Expand dim to apply broadcasting.
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / union_area

    bc_mins = tf.minimum(b1_mins, b2_mins)
    bc_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(bc_maxes - bc_mins, 0.)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    giou = iou - (enclose_area - union_area) / enclose_area
    return giou

if tf.version.VERSION.startswith('1.'):

    def YoloLoss(y_true,
                 yolo_output,
                 idx,
                 ignore_thresh: float = .5,
                 box_loss=BOX_LOSS.GIOU,
                 print_loss: bool = False):
        '''Return yolo_loss tensor

        Parameters
        ----------
        yolo_output: the output of yolo_body
        y_true: the output of preprocess_true_boxes
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss

        Returns
        -------
        loss: tensor, shape=(1,)
        '''

        grid_steps = [16, 8]
        grid_step = grid_steps[idx]
        #anchor_mask = [[3, 4, 5], [0, 1, 2]]
        loss = 0
        m = tf.shape(yolo_output)[0]  # batch size, tensor
        mf = tf.cast(m, yolo_output.dtype)
        #print("------------------------------")
        #print(m,mf)
        #print("------------------------------")
        object_mask = y_true[..., 2:3]
        true_class_probs = y_true[..., 3:]
        input_shape = tf.shape(yolo_output)[1:3] * grid_step
        
        

        grid, pred_xy, box_confidence = yolo_head(
            yolo_output, input_shape, calc_loss=True)


        
        
        confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                 logits=yolo_output[..., 2:3])

        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=true_class_probs, logits=yolo_output[..., 3:])
        confidence_loss = tf.reduce_sum(confidence_loss) / mf
        class_loss = tf.reduce_sum(class_loss) / mf



        grid_shape = tf.cast(tf.shape(yolo_output)[1:3], y_true.dtype)
        raw_true_xy = y_true[..., :2] * grid_shape[::-1] - grid
        
        xy_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=raw_true_xy, logits=yolo_output[..., 0:2])
        
        xy_loss = tf.reduce_sum(xy_loss) / mf
        loss += xy_loss + confidence_loss + class_loss
        if print_loss:
            tf.print(loss, xy_loss, confidence_loss, class_loss)
        return loss
else:
    class YoloLoss(tf.keras.losses.Loss):

        def __init__(self,
                     idx,
                     anchors,
                     ignore_thresh=.5,
                     box_loss=BOX_LOSS.GIOU,
                     print_loss=True):
            super(YoloLoss, self).__init__(reduction=tf.losses.Reduction.NONE,name='yolo_loss')
            grid_steps = [32, 16, 8]
            anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            self.idx = idx
            self.ignore_thresh = ignore_thresh
            self.box_loss = box_loss
            self.print_loss = print_loss
            self.grid_step = grid_steps[self.idx]
            self.anchor = anchors[anchor_masks[idx]]

        def call(self, y_true, yolo_output):
            loss = 0
            m = tf.shape(yolo_output)[0]  # batch size, tensor
            mf = tf.cast(m, yolo_output.dtype)
            object_mask = y_true[..., 4:5]
            true_class_probs = y_true[..., 5:]
            input_shape = tf.shape(yolo_output)[1:3] * self.grid_step
            grid, pred_xy, pred_wh, box_confidence = yolo_head(yolo_output,
                                                               self.anchor,
                                                               input_shape,
                                                               calc_loss=True)
            pred_box = tf.concat([pred_xy, pred_wh], -1)
            # Find ignore mask, iterate over each of batch.
            object_mask_bool = tf.cast(object_mask, 'bool')

            true_box = tf.boolean_mask(y_true[..., 0:4], object_mask_bool[..., 0])
            iou = box_iou(tf.expand_dims(pred_box, -2), tf.expand_dims(true_box, 0))
            best_iou = tf.reduce_max(iou, axis=-1)
            ignore_mask = tf.cast(best_iou < self.ignore_thresh, true_box.dtype)

            ignore_mask = tf.expand_dims(ignore_mask, -1)
            confidence_loss = (object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                     logits=yolo_output[..., 4:5]) + \
                               (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                           logits=yolo_output[...,
                                                                                                  4:5]) * ignore_mask)
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_class_probs, logits=yolo_output[..., 5:])
            confidence_loss = tf.reduce_sum(confidence_loss) / mf
            class_loss = tf.reduce_sum(class_loss) / mf

            if self.box_loss == BOX_LOSS.GIOU:
                giou = box_giou(pred_box[..., :4], y_true[..., :4])
                giou_loss = object_mask * (1 - tf.expand_dims(giou, -1))
                giou_loss = tf.reduce_sum(giou_loss) / mf
                loss += giou_loss + confidence_loss + class_loss
                if self.print_loss:
                    tf.print(
                        str(self.idx) + ':', giou_loss, confidence_loss, class_loss,
                        tf.reduce_sum(ignore_mask))
            elif self.box_loss == BOX_LOSS.MSE:
                grid_shape = tf.cast(tf.shape(yolo_output)[1:3], y_true.dtype)
                raw_true_xy = y_true[..., :2] * grid_shape[::-1] - grid
                raw_true_wh = tf.math.log(y_true[..., 2:4] / self.anchor *
                                          input_shape[::-1])
                raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh,
                                                      tf.zeros_like(raw_true_wh))
                box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]
                xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=raw_true_xy, logits=yolo_output[..., 0:2])
                wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(
                    raw_true_wh - yolo_output[..., 2:4])
                xy_loss = tf.reduce_sum(xy_loss) / mf
                wh_loss = tf.reduce_sum(wh_loss) / mf
                loss += xy_loss + wh_loss + confidence_loss + class_loss
                if self.print_loss:
                    tf.print(loss, xy_loss, wh_loss, confidence_loss, class_loss,
                             tf.reduce_sum(ignore_mask))
            return loss
