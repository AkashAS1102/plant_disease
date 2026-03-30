"""
Plant Disease Recognition Model Training Script
================================================
This script automatically downloads the PlantVillage dataset and trains
a CNN model for plant disease classification.

Requirements:
    pip install tensorflow tensorflow-datasets pillow numpy
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print("=" * 60)
print("Plant Disease Recognition Model Training")
print("=" * 60)

print("\n[1/6] Loading PlantVillage dataset from TensorFlow Datasets...")
print("This may take a few minutes on first run (downloading ~870MB)")

dataset, dataset_info = tfds.load(
    'plant_village',
    split='train',
    as_supervised=True,
    with_info=True
)

num_classes = dataset_info.features['label'].num_classes
class_names = dataset_info.features['label'].names

print(f"\n✓ Dataset loaded successfully!")
print(f"  Total classes: {num_classes}")
print(f"  Sample classes: {class_names[:5]}")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

print("\n[2/6] Preprocessing dataset...")

def preprocess_image(image, label):
    """Resize and normalize image to [0, 1] range"""
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)

train_size = int(0.8 * dataset_info.splits['train'].num_examples)
val_size = dataset_info.splits['train'].num_examples - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

print(f"  Training samples: {train_size}")
print(f"  Validation samples: {val_size}")

print("\n[3/6] Applying data augmentation...")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def augment_image(image, label):
    """Apply data augmentation to training images"""
    return data_augmentation(image, training=True), label

train_dataset = train_dataset.map(augment_image, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

print("  ✓ Augmentation layers added (flip, rotation, zoom, contrast)")

print("\n[4/6] Building Convolutional Neural Network...")

model = keras.Sequential([
    layers.Input(shape=(256, 256, 3)),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "=" * 60)
model.summary()
print("=" * 60)

print("\n[5/6] Training model (3 epochs for quick execution)...")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n[6/6] Saving trained model...")

model.save('plant_disease_model.h5')

print("\n" + "=" * 60)
print("✓ SUCCESS! Model saved as 'plant_disease_model.h5'")
print("=" * 60)

val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
print(f"\nFinal Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Final Validation Loss: {val_loss:.4f}")

print("\n🎉 Training complete! You can now run the FastAPI backend.")
print("   Command: uvicorn main:app --reload")
