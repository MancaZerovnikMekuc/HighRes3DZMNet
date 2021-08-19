# -*- coding: utf-8 -*-
import tensorflow as tf
class MeanZero():
    """Mean Zero weight constraint.
    Constrains the weights to have a mean zero value
    # Arguments
        axis: integer, axis along which to calculate the mean.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,

            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
    """

    def __init__(self, axis=None):
        self.axis = axis

    def __call__(self, w):
        mean=tf.reduce_mean(w,axis=self.axis,keepdims=True)
        w = w - mean
        return w

    def get_config(self):
        return {'axis': self.axis}