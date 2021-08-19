# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.deconvolution import DeconvLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet
from niftynet.utilities.util_common import look_up_operations
from niftynet.io.misc_io import image3_axial
from niftynet.evaluation import segmentation_evaluations

class VNetBn(BaseNet):
    """
    implementation of V-Net:
      Milletari et al., "V-Net: Fully convolutional neural networks for
      volumetric medical image segmentation", 3DV '16
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='VNet'):

        super(VNetBn, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.n_features = [16, 32, 64, 128, 256]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        assert layer_util.check_spatial_dims(images, lambda x: x % 8 == 0)

        if layer_util.infer_spatial_rank(images) == 2:
            padded_images = tf.tile(images, [1, 1, 1, self.n_features[0]])
        elif layer_util.infer_spatial_rank(images) == 3:
            padded_images = tf.tile(images, [1, 1, 1, 1, self.n_features[0]])
        else:
            raise ValueError('not supported spatial rank of the input image')
        # downsampling  blocks
        res_1, down_1 = VNetBlock('DOWNSAMPLE', 1,
                                  self.n_features[0],
                                  self.n_features[1],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  k_constraint=False,
                                  name='L1')(images, padded_images)
        res_2, down_2 = VNetBlock('DOWNSAMPLE', 2,
                                  self.n_features[1],
                                  self.n_features[2],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='L2')(down_1, down_1)
        res_3, down_3 = VNetBlock('DOWNSAMPLE', 3,
                                  self.n_features[2],
                                  self.n_features[3],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='L3')(down_2, down_2)
        res_4, down_4 = VNetBlock('DOWNSAMPLE', 3,
                                  self.n_features[3],
                                  self.n_features[4],
                                  acti_func=self.acti_func,
                                  name='L4')(down_3, down_3)
        # upsampling blocks
        _, up_4 = VNetBlock('UPSAMPLE', 3,
                            self.n_features[4],
                            self.n_features[4],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='V_')(down_4, down_4)
        concat_r4 = ElementwiseLayer('CONCAT')(up_4, res_4)
        _, up_3 = VNetBlock('UPSAMPLE', 3,
                            self.n_features[4],
                            self.n_features[3],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='R4')(concat_r4, up_4)
        concat_r3 = ElementwiseLayer('CONCAT')(up_3, res_3)
        _, up_2 = VNetBlock('UPSAMPLE', 3,
                            self.n_features[3],
                            self.n_features[2],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='R3')(concat_r3, up_3)
        concat_r2 = ElementwiseLayer('CONCAT')(up_2, res_2)
        _, up_1 = VNetBlock('UPSAMPLE', 2,
                            self.n_features[2],
                            self.n_features[1],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='R2')(concat_r2, up_2)
        # final class score
        concat_r1 = ElementwiseLayer('CONCAT')(up_1, res_1)
        _, output_tensor = VNetBlock('SAME', 1,
                                     self.n_features[1],
                                     self.num_classes,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func=self.acti_func,
                                     name='R1')(concat_r1, up_1)


        # Segmentation results
        """
        seg_argmax = tf.to_float(tf.expand_dims(tf.argmax(output_tensor, -1), -1))
        seg_summary = seg_argmax * (255. / self.num_classes - 1)

        # Image Summary
        n_spatial_dims = 3
        input_tensor = images
        norm_axes = list(range(1, n_spatial_dims + 1))
        mean, var = tf.nn.moments(input_tensor, axes=norm_axes, keep_dims=True)
        timg = tf.to_float(input_tensor - mean) / (tf.sqrt(var) * 2.)
        timg = (timg + 1.) * 127.
        single_channel = tf.reduce_mean(timg, -1, True)
        img_summary = tf.minimum(255., tf.maximum(0., single_channel))
        if n_spatial_dims == 2:
            tf.summary.image(
                tf.get_default_graph().unique_name('imgseg'),
                tf.concat([img_summary, seg_summary], 1),
                20, [tf.GraphKeys.SUMMARIES])
        elif n_spatial_dims == 3:
            # Show summaries
            image3_axial(
                tf.get_default_graph().unique_name('imgseg'),
                tf.concat([img_summary, seg_summary], 1),
                20, [tf.GraphKeys.SUMMARIES])
        else:
            raise NotImplementedError(
                'Image Summary only supports 2D and 3D images')


        """
        return output_tensor


SUPPORTED_OP = set(['DOWNSAMPLE', 'UPSAMPLE', 'SAME'])


class VNetBlock(TrainableLayer):
    def __init__(self,
                 func,
                 n_conv,
                 n_feature_chns,
                 n_output_chns,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='vnet_block',
                 k_constraint=None):

        super(VNetBlock, self).__init__(name=name)

        self.func = look_up_operations(func.upper(), SUPPORTED_OP)
        self.n_conv = n_conv
        self.n_feature_chns = n_feature_chns
        self.n_output_chns = n_output_chns
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}
        self.constraint = k_constraint

    def layer_op(self, main_flow, bypass_flow):
        for i in range(self.n_conv):
            main_flow = ConvolutionalLayer(name='conv_{}'.format(i),
                                  n_output_chns=self.n_feature_chns,
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  kernel_size=5, k_constraint=self.constraint)(main_flow)
            if i < self.n_conv - 1:  # no activation for the last conv layer
                main_flow = ActiLayer(
                    func=self.acti_func,
                    regularizer=self.regularizers['w'])(main_flow)
        res_flow = ElementwiseLayer('SUM')(main_flow, bypass_flow)

        if self.func == 'DOWNSAMPLE':
            main_flow = ConvolutionalLayer(
                name='downsample',
                n_output_chns=self.n_output_chns,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                kernel_size=2, stride=2, with_bias=True, k_constraint=self.constraint)(res_flow)
        elif self.func == 'UPSAMPLE':
            main_flow = DeconvolutionalLayer(
                name='upsample',
                n_output_chns=self.n_output_chns,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                kernel_size=2, stride=2, with_bias=True)(res_flow)
        elif self.func == 'SAME':
            main_flow = ConvolutionalLayer(name='conv_1x1x1',
                                  n_output_chns=self.n_output_chns,
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  b_initializer=self.initializers['b'],
                                  b_regularizer=self.regularizers['b'],
                                  kernel_size=1, with_bias=True, k_constraint=self.constraint)(res_flow)
        main_flow = ActiLayer(self.acti_func)(main_flow)
        print(self)
        return res_flow, main_flow
