import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2 # <-- Our new import


# --- PATH 1 FUNCTION ---
def build_custom_cnn(input_shape):
    """
    Builds the custom CNN model (Path 1).
    """
    
    # Define the augmentation layers
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.GaussianNoise(0.01)
    ], name="data_augmentation")

    # Define the model
    model = Sequential([
        data_augmentation,
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name="custom_cnn_model")
    
    return model


# --- PATH 2 FUNCTION ---
def build_transfer_model(input_shape=(224, 224, 3)):
    """
    Builds the transfer learning model (Path 2).
    """
    
    # 1. Define augmentation (same as before)
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.GaussianNoise(0.01)
    ], name="data_augmentation")

    # 2. Load the pre-trained base model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 3. Freeze the base model (this was our best-performing setup)
    base_model.trainable = False

    # 4. Build the full model
    model = Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name="transfer_learning_model")
    
    return model