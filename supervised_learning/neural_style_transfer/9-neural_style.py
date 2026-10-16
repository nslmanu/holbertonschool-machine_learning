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
        self.generate_features()

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
            if all(name in outputs for name in selected):
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

    def generate_features(self):
        """Extract and store the reference style and content features."""
        style_input = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        content_input = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )
        style_outputs = self.model(style_input)
        content_outputs = self.model(content_input)

        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Return the mean squared Gram difference for one style layer.

        Args:
            style_output: Generated image features of shape (1, h, w, c).
            gram_target: Reference Gram tensor of shape (1, c, c).

        Returns:
            Scalar tensor measuring the layer's style mismatch.
        """
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or style_output.shape.rank != 4):
            raise TypeError('style_output must be a tensor of rank 4')

        channels = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, channels, channels)):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'.format(
                    channels, channels
                )
            )

        gram_generated = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_generated - gram_target))

    def style_cost(self, style_outputs):
        """Return the equally weighted cost across all style layers.

        Args:
            style_outputs: List of generated features, one per style layer.

        Returns:
            Scalar tensor containing the average layer style cost.
        """
        layer_count = len(self.style_layers)
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != layer_count):
            raise TypeError(
                'style_outputs must be a list with a length of {}'.format(
                    layer_count
                )
            )

        costs = [
            self.layer_style_cost(output, target)
            for output, target in zip(
                style_outputs, self.gram_style_features
            )
        ]
        return tf.add_n(costs) / layer_count

    def content_cost(self, content_output):
        """Measure the mean squared difference from reference content.

        Args:
            content_output: Generated content tensor matching the reference.

        Returns:
            Scalar tensor containing the content cost.
        """
        target_shape = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or content_output.shape != target_shape):
            raise TypeError(
                'content_output must be a tensor of shape {}'.format(
                    target_shape
                )
            )

        return tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

    def total_cost(self, generated_image):
        """Return total, content, and style costs for a generated image.

        Args:
            generated_image: Tensor matching the reference image shape.

        Returns:
            Tuple (total cost, unweighted content cost, unweighted style cost).
        """
        target_shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != target_shape):
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(
                    target_shape
                )
            )

        model_input = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )
        outputs = self.model(model_input)
        J_content = self.content_cost(outputs[-1])
        J_style = self.style_cost(outputs[:-1])
        J = self.alpha * J_content + self.beta * J_style
        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """Return image gradients and total, content, and style costs.

        Args:
            generated_image: Tensor or variable matching content_image.

        Returns:
            Tuple (gradients, total cost, content cost, style cost).
        """
        target_shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != target_shape):
            raise TypeError(
                'generated_image must be a tensor of shape {}'.format(
                    target_shape
                )
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)
        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Optimize image pixels with Adam and return the best image and cost.

        Args:
            iterations: Positive number of optimization updates.
            step: Optional reporting interval, smaller than iterations.
            lr: Positive learning rate.
            beta1: Adam first moment decay, between zero and one.
            beta2: Adam second moment decay, between zero and one.

        Returns:
            Best image as a NumPy array (h, w, 3), and its scalar cost.
        """
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be positive')
        if step is not None:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if step <= 0 or step >= iterations:
                raise ValueError('step must be positive and less than '
                                 'iterations')
        if type(lr) not in (int, float):
            raise TypeError('lr must be a number')
        if not lr > 0:
            raise ValueError('lr must be positive')
        if type(beta1) is not float:
            raise TypeError('beta1 must be a float')
        if not 0 <= beta1 <= 1:
            raise ValueError('beta1 must be in the range [0, 1]')
        if type(beta2) is not float:
            raise TypeError('beta2 must be a float')
        if not 0 <= beta2 <= 1:
            raise ValueError('beta2 must be in the range [0, 1]')

        generated_image = tf.Variable(self.content_image, trainable=True)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta1, beta_2=beta2
        )
        best_cost = float('inf')
        best_image = generated_image.numpy()[0].copy()

        for i in range(iterations + 1):
            gradients, total, content, style = self.compute_grads(
                generated_image
            )
            cost = float(total.numpy())
            if cost < best_cost:
                best_cost = cost
                best_image = generated_image.numpy()[0].copy()

            if step is not None and (i % step == 0 or i == iterations):
                print('Cost at iteration {}: {}, content {}, style {}'.format(
                    i, cost, float(content.numpy()), float(style.numpy())
                ))

            if i < iterations:
                optimizer.apply_gradients([(gradients, generated_image)])
                generated_image.assign(
                    tf.clip_by_value(generated_image, 0.0, 1.0)
                )

        return best_image, best_cost
