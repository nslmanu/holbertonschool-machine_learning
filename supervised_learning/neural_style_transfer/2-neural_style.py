#!/usr/bin/env python3
"""Prepare content and style images for neural style transfer."""

import numpy as np
import tensorflow as tf


class NST:
    """Store reference images and neural style transfer settings."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Validate references and weights, then prepare both images."""
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3 or style_image.shape[-1] != 3):
            raise TypeError('style_image must be a numpy.ndarray '
                            'with shape (h, w, 3)')
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3 or content_image.shape[-1] != 3):
            raise TypeError('content_image must be a numpy.ndarray '
                            'with shape (h, w, 3)')
        if not isinstance(alpha, (int, float)) or not alpha >= 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or not beta >= 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Return a normalized batch tensor with longest side 512 pixels."""
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3 or image.shape[-1] != 3):
            raise TypeError('image must be a numpy.ndarray '
                            'with shape (h, w, 3)')

        height, width = image.shape[:2]
        scale = 512 / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized = tf.image.resize(
            image, (new_height, new_width), method='bicubic'
        )
        normalized = tf.clip_by_value(resized / 255.0, 0.0, 1.0)
        return tf.expand_dims(normalized, axis=0)

    def load_model(self):
        """Load frozen VGG19 features, replacing max pooling with averages."""
        base = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        inputs = base.input
        features = inputs
        outputs = {}
        selected = self.style_layers + [self.content_layer]

        for layer in base.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                features = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding, data_format=layer.data_format,
                    name=layer.name
                )(features)
            else:
                layer.trainable = False
                features = layer(features)
            if layer.name in selected:
                outputs[layer.name] = features
            if layer.name == self.content_layer:
                break

        self.model = tf.keras.Model(
            inputs=inputs, outputs=[outputs[name] for name in selected]
        )
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Compute spatially normalized channel products for a rank-4 tensor.

        Args:
            input_layer: Tensor or variable of shape (1, h, w, c).

        Returns:
            Tensor of shape (1, c, c) containing the Gram matrix.
        """
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or input_layer.shape.rank != 4):
            raise TypeError('input_layer must be a tensor of rank 4')

        shape = tf.shape(input_layer)
        features = tf.reshape(
            input_layer, [shape[0], shape[1] * shape[2], shape[3]]
        )
        gram = tf.matmul(features, features, transpose_a=True)
        spatial_size = tf.cast(shape[1] * shape[2], gram.dtype)
        return gram / spatial_size
