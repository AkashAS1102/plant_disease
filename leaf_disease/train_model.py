import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, applications
import json
import os

def main():
    print("Loading PlantVillage dataset with Transfer Learning configurations...")
    
    # We will split the single 'train' split into 80% training and 20% validation
    # to prove the model's accuracy on unseen data
    # We explicitly define data_dir to a shorter path to bypass Windows 260-char path limit during extraction
    (ds_train, ds_val), ds_info = tfds.load(
        'plant_village', 
        split=['train[:80%]', 'train[80%:]'], 
        as_supervised=True, 
        with_info=True,
        data_dir='C:/T'
    )
    
    # Extract EXACT string class labels and save them to a JSON file for the FastAPI backend
    class_names = ds_info.features['label'].names
    num_classes = len(class_names)
    print(f"Dataset extracted. Found {num_classes} classes.")
    
    with open('class_indices.json', 'w') as f:
        json.dump(class_names, f)
    print("Class names mapping saved to class_indices.json")

    # MobileNetV2 Optimal Configs
    IMG_SIZE = 224 # Standard for MobileNet architectures
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE

    def format_image(image, label):
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image, label

    print("Configuring robust data pipelines...")
    
    # Training Pipeline
    ds_train = ds_train.map(format_image, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=1000)
    ds_train = ds_train.batch(BATCH_SIZE)
    ds_train = ds_train.prefetch(buffer_size=AUTOTUNE)

    # Validation Pipeline
    ds_val = ds_val.map(format_image, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.batch(BATCH_SIZE)
    ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation Block (Prevent Overfitting)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    print("Building Transfer Learning Architecture using MobileNetV2...")
    
    # Create the base model from the pre-trained model MobileNetV2
    # MobileNetV2 requires pixels to be scaled between [-1, 1], not [0, 1]
    base_model = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the pre-trained weights so we don't accidentally wreck them during initial training
    base_model.trainable = False

    # Build the final classification pipeline
    model = models.Sequential([
        layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # 1. Augment images
        data_augmentation,
        
        # 2. Rescale input to [-1, 1] required for MobileNetV2
        # (Pixel / 127.5) - 1 maps [0, 255] to [-1.0, 1.0]
        layers.Rescaling(1./127.5, offset=-1),
        
        # 3. Base MobileNetV2 feature extractor
        base_model,
        
        # 4. Global Average Pooling (Convert 2D features to 1D vector)
        layers.GlobalAveragePooling2D(),
        
        # 5. Dropout to reduce overfitting
        layers.Dropout(0.2),
        
        # 6. Final Classification Output with Softmax
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    # We use sparse categorical crossentropy because labels are integers (0, 1, ..., 37)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Model parameters for demonstration
    EPOCHS = 5

    print(f"Starting training for {EPOCHS} epochs to achieve >90% validation accuracy...")
    
    # Train the model while observing validation metrics instantly
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val
    )

    # Save the compiled and trained model
    model_save_path = "plant_disease_model.h5"
    model.save(model_save_path)
    print(f"Transfer Learning Model successfully saved to {model_save_path}. Validation accuracy should be pristine.")

if __name__ == "__main__":
    main()
