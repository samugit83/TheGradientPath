#!/usr/bin/env python3
"""
A Keras Functional API example with TensorBoard integration:
- loads MNIST
- builds a model with Dense, Dropout, and BatchNormalization
- logs graph, metrics, histograms, and high-level architecture image (Visualkeras) to TensorBoard
- trains and evaluates the model
- predicts and saves sample outputs
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Flatten,
)
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import visualkeras


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return (x_train, y_train), (x_test, y_test)


def build_model(input_shape=(28, 28), num_classes=10):
    inputs = Input(shape=input_shape, name="mnist_input")
    x = Flatten(name="flatten")(inputs)
    x = Dense(128, name="dense_1")(x)
    x = Activation("relu", name="act_1")(x)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.5, name="dropout_1")(x)
    x = Dense(64, name="dense_2")(x)
    x = Activation("relu", name="act_2")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.5, name="dropout_2")(x)
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    return Model(inputs=inputs, outputs=outputs, name="mnist_mlp")


def predict_and_save_samples(model, x_test, y_test, num_samples=5, output_dir="predictions"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sample_indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    images = x_test[sample_indices]
    labels = y_test[sample_indices]
    preds = model.predict(images)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels, axis=1)
    print(f"\n--- Predictions on {num_samples} samples ---")
    for i, img in enumerate(images):
        print(f"Sample {i+1}: Predicted={pred_classes[i]}, True={true_classes[i]}")
        img_plot = img.reshape(28,28)
        path = os.path.join(output_dir, f"sample_{i+1}_true_{true_classes[i]}_pred_{pred_classes[i]}.png")
        plt.imsave(path, img_plot, cmap='gray')
        print(f"  Saved image to: {path}")
    print("-----------------------------------------")


def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Build and compile model
    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Prepare TensorBoard log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("logs", "graphs", timestamp)
    os.makedirs(logdir, exist_ok=True)

    # Generate high-level architecture using Visualkeras
    arch_path = os.path.join(logdir, 'model_visualkeras.png')
    # Create a layered view; adjust scale and spacing as needed
    visualkeras.layered_view(
        model,
        to_file=arch_path,
        legend=True,
        draw_volume=False,
        scale_xy=1.5,
        scale_z=1,
        spacing=20
    )

    # Log Visualkeras image to TensorBoard
    with tf.summary.create_file_writer(logdir).as_default():
        img = tf.io.read_file(arch_path)
        img = tf.image.decode_png(img, channels=4)
        tf.summary.image("Model Visualization", tf.expand_dims(img, 0), step=0)

    # TensorBoard callback: logs scalars, histograms, images, and profile for Graphs
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=1
    )

    print(f"TensorBoard logs in: {os.path.abspath(logdir)}")
    print("Run: tensorboard --logdir logs")

    # Train model
    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        callbacks=[tensorboard_cb],
        verbose=2
    )

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}, loss: {loss:.4f}")

    # Predict and save sample outputs
    predict_and_save_samples(model, x_test, y_test)


if __name__ == "__main__":
    main()
