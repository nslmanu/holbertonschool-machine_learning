#!/usr/bin/env python3
"""Train an ImageNet-pretrained EfficientNetB0 on CIFAR-10.

Run with Python 3.9 and TensorFlow 2.15 (plus numpy and h5py).
The first run downloads CIFAR-10 and ImageNet weights. A GPU is recommended.
Outputs in the working directory: cifar10.h5, training_head.csv,
training_finetune.csv (when needed), and training_results.json.

Resizing is a serializable alternative to the suggested Lambda layer:
the saved model loads without custom_objects or external resize functions.
Accuracy is measured by the script, not guaranteed before training.
"""

import json

import numpy as np
import tensorflow as tf
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Return float32 images and one-hot CIFAR-10 labels.

    Args:
        X: Array of raw images with shape (m, 32, 32, 3), range 0-255.
        Y: Integer labels with shape (m,) or (m, 1).

    Returns:
        X_p, Y_p: NumPy arrays with shapes (m, 32, 32, 3) and (m, 10).

    EfficientNet includes normalization, so do not divide pixels by 255.
    The complete model handles spatial resizing internally.
    """
    X_p = X.astype('float32')
    Y_p = K.utils.to_categorical(Y.reshape(-1), num_classes=10)
    return X_p, Y_p


def compile_model(model, learning_rate):
    """Compile a model for classification with one-hot labels."""
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


def callbacks(logfile, patience):
    """Log each epoch and restore weights with the best validation score."""
    return [
        K.callbacks.CSVLogger(logfile),
        K.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7
        ),
        K.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=patience,
            restore_best_weights=True
        ),
        K.callbacks.TerminateOnNaN()
    ]


def build_models():
    """Build a frozen feature extractor, classifier, and complete model."""
    base = K.applications.EfficientNetB0(
        include_top=False, weights='imagenet',
        input_shape=(224, 224, 3), pooling='avg'
    )
    base.trainable = False

    inputs = K.Input(shape=(32, 32, 3))
    resized = K.layers.Resizing(
        224, 224, interpolation='bilinear', name='resize_images'
    )(inputs)
    # Keep BatchNormalization statistics fixed, including during fine-tuning.
    features = base(resized, training=False)
    extractor = K.Model(inputs, features, name='feature_extractor')

    head = K.Sequential([
        K.Input(shape=(base.output_shape[-1],)),
        K.layers.Dense(
            256, activation='relu',
            kernel_initializer=K.initializers.HeNormal(seed=0),
            kernel_regularizer=K.regularizers.l2(1e-4)
        ),
        K.layers.Dropout(0.3),
        K.layers.Dense(10, activation='softmax')
    ], name='classifier')
    model = K.Model(inputs, head(features), name='cifar10_transfer')
    return base, extractor, head, model


def augment(image, label):
    """Apply label-preserving CIFAR-10 augmentation for fine-tuning."""
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, (32, 32, 3))
    image = tf.image.random_flip_left_right(image)
    return image, label


def main():
    """Train, save, reload, and evaluate the complete CIFAR-10 model."""
    K.utils.set_random_seed(0)
    (X, Y), (X_test, Y_test) = K.datasets.cifar10.load_data()

    # Split the training set by class: 45,000 train and 5,000 validation.
    rng = np.random.default_rng(0)
    training_ids = []
    validation_ids = []
    for label in range(10):
        indices = np.flatnonzero(Y.reshape(-1) == label)
        rng.shuffle(indices)
        validation_ids.extend(indices[:500])
        training_ids.extend(indices[500:])
    rng.shuffle(training_ids)
    rng.shuffle(validation_ids)
    X_train, Y_train = preprocess_data(X[training_ids], Y[training_ids])
    X_val, Y_val = preprocess_data(X[validation_ids], Y[validation_ids])
    del X, Y

    base, extractor, head, model = build_models()
    print('Extracting frozen features once per image...')
    train_features = extractor.predict(X_train, batch_size=32, verbose=1)
    val_features = extractor.predict(X_val, batch_size=32, verbose=1)

    compile_model(head, 1e-3)
    head.fit(
        train_features, Y_train,
        validation_data=(val_features, Y_val),
        epochs=40, batch_size=256, shuffle=True,
        callbacks=callbacks('training_head.csv', patience=7), verbose=2
    )
    _, best_accuracy = head.evaluate(val_features, Y_val, verbose=0)
    del train_features, val_features

    compile_model(model, 1e-3)
    model.save('cifar10.h5')
    results = {
        'backbone': 'EfficientNetB0',
        'image_size': 224,
        'seed': 0,
        'training_examples': len(X_train),
        'validation_examples': len(X_val),
        'head_validation_accuracy': float(best_accuracy),
        'selected_stage': 'frozen_features'
    }

    # Aim above the required score while retaining the best saved model.
    if best_accuracy < 0.90:
        print('Fine-tuning the final layers at a low learning rate...')
        base.trainable = True
        for layer in base.layers:
            layer.trainable = False
        for layer in base.layers[-30:]:
            if not isinstance(layer, K.layers.BatchNormalization):
                layer.trainable = True
        compile_model(model, 1e-5)

        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        dataset = dataset.shuffle(10000, seed=0)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        validation = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        validation = validation.batch(32).prefetch(tf.data.AUTOTUNE)
        model.fit(
            dataset, validation_data=validation, epochs=15,
            callbacks=callbacks('training_finetune.csv', patience=5),
            verbose=2
        )
        _, accuracy = model.evaluate(validation, verbose=0)
        results['finetune_validation_accuracy'] = float(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            results['selected_stage'] = 'fine_tuning'
            model.save('cifar10.h5')

    # Match the evaluator: load a compiled file with no custom objects.
    saved_model = K.models.load_model('cifar10.h5')
    _, validation_accuracy = saved_model.evaluate(
        X_val, Y_val, batch_size=32, verbose=0
    )
    del X_train, Y_train, X_val, Y_val
    X_test, Y_test = preprocess_data(X_test, Y_test)
    test_loss, test_accuracy = saved_model.evaluate(
        X_test, Y_test, batch_size=32, verbose=1
    )
    results['saved_validation_accuracy'] = float(validation_accuracy)
    results['test_accuracy'] = float(test_accuracy)
    results['test_loss'] = float(test_loss)
    results['validation_target_met'] = bool(validation_accuracy >= 0.87)
    with open('training_results.json', 'w', encoding='utf-8') as logfile:
        json.dump(results, logfile, indent=2)
        logfile.write('\n')
    print('Saved compiled model: cifar10.h5')
    print('Validation accuracy: {:.2%}'.format(validation_accuracy))
    print('Test accuracy: {:.2%}'.format(test_accuracy))
    if validation_accuracy < 0.87 or test_accuracy < 0.87:
        print('87% was not reached on both sets; further work is needed.')


if __name__ == '__main__':
    main()
