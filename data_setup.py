import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(dataset_path, img_size, batch_size, color_mode):
    """
    Creates training and validation data generators.
    """
    
    base_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # Use 20% of data for validation
    )

    print("Creating training data generator...")
    train_gen = base_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'  # Specify this is the training subset
    )

    print("Creating validation data generator...")
    val_gen = base_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        color_mode=color_mode,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'  # Specify this is the validation subset
    )
    
    return train_gen, val_gen