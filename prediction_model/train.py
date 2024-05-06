import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def train_pneumonia_model():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        "./x-ray_data/train",
        target_size=(150, 150),
        batch_size=32,
        class_mode="sparse",
        subset="training",
    )

    validation_generator = train_datagen.flow_from_directory(
        "./x-ray_data/val",
        target_size=(150, 150),
        batch_size=32,
        class_mode="sparse",
        subset="validation",
    )

    test_generator = train_datagen.flow_from_directory(
        "./x-ray_data/test",
        target_size=(150, 150),
        batch_size=32,
        class_mode="sparse",
    )

    class_weights = dict(
        zip(
            np.unique(train_generator.classes),
            (
                len(train_generator.classes)
                / (
                    len(np.unique(train_generator.classes))
                    * np.bincount(train_generator.classes)
                )
            ),
        )
    )

    print(class_weights)

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    test_acc = model.evaluate(test_generator)
    print("Test accuracy:", test_acc[1])

    model.save("pneumonia_prediction.keras")


def train_lungs_model():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        "./lung_validation_data/train",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        subset="training",
    )

    validation_generator = train_datagen.flow_from_directory(
        "./lung_validation_data/val",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
        subset="validation",
    )

    test_generator = train_datagen.flow_from_directory(
        "./lung_validation_data/test",
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary",
    )

    class_weights = dict(
        zip(
            np.unique(train_generator.classes),
            (
                len(train_generator.classes)
                / (
                    len(np.unique(train_generator.classes))
                    * np.bincount(train_generator.classes)
                )
            ),
        )
    )

    print(class_weights)

    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        weighted_metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        class_weight=class_weights,
    )

    test_acc = model.evaluate(test_generator)
    print("Test accuracy:", test_acc[1])

    model.save("lungs_prediction.keras")
