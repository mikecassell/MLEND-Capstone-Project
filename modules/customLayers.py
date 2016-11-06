# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from math import ceil

import tflearn
from tflearn import variables as vs
from tflearn import activations
from tflearn import initializations
from tflearn import losses
from tflearn import utils

from tflearn import batch_normalization




def conv_2d_BN(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=True, weights_init='xavier',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, reuse=False, scope=None,
            name="Conv2D", batch_norm=False):
    """ Convolution 2D.
    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, new height, new width, nb_filter].
    Arguments:
        incoming: `Tensor`. Incoming 4-D Tensor.
        nb_filter: `int`. The number of convolutional filters.
        filter_size: `int` or `list of int`. Size of filters.
        strides: 'int` or list of `int`. Strides of conv operation.
            Default: [1 1 1 1].
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'Conv2D'.
        batch_norm: If true, add batch normalization with default TFLearn 
            parameters before the activation layer 
    Attributes:
        scope: `Scope`. This layer scope.
        W: `Variable`. Variable representing filter weights.
        b: `Variable`. Variable representing biases.
    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)

        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = tf.nn.conv2d(incoming, W, strides, padding)
        if b: inference = tf.nn.bias_add(inference, b)

        if batch_norm:
            inference = batch_normalization(inference)
        
        if isinstance(activation, str):
            if activation == 'softmax':
                shapes = inference.get_shape()

                inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference

def fully_connected_BN(incoming, n_units, activation='linear', bias=True,
                    weights_init='truncated_normal', bias_init='zeros',
                    regularizer=None, weight_decay=0.001, trainable=True,
                    restore=True, reuse=False, scope=None,
                    name="FullyConnected", batch_norm=False):
    """ Fully Connected.
    A fully connected layer.
    Input:
        (2+)-D Tensor [samples, input dim]. If not 2D, input will be flatten.
    Output:
        2D Tensor [samples, n_units].
    Arguments:
        incoming: `Tensor`. Incoming (2+)D Tensor.
        n_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'truncated_normal'.
        bias_init: `str` (name) or `Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'FullyConnected'.
        batch_norm: if True add a batch normalization layer before the activation 
            function.
    Attributes:
        scope: `Scope`. This layer scope.
        W: `Tensor`. Variable representing units weights.
        b: `Tensor`. Variable representing biases.
    """
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    # Build variables and inference.
    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=[n_inputs, n_units], regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=[n_units], initializer=bias_init,
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 2:
            inference = tf.reshape(inference, [-1, n_inputs])

        inference = tf.matmul(inference, W)
        if batch_norm:
            inference = batch_normalization(inference)
            
        if b: inference = tf.nn.bias_add(inference, b)

        if isinstance(activation, str):
            inference = activations.get(activation)(inference)
        elif hasattr(activation, '__call__'):
            inference = activation(inference)
        else:
            raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    #tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference