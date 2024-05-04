import tensorflow as tf
from tensorflow.keras import layers, models

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

print(train_generator.class_indices)

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
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

model.save("pneumonia_prediction.keras")

test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)
