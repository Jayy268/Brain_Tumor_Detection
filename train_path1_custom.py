import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import our custom functions from the .py files
from data_setup import create_data_generators
from model_builder import build_custom_cnn

# --- 1. Set Parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = '/content/drive/My Drive/Computer Vision Project/datasets/brain_tumor_dataset'
COLOR_MODE_P1 = 'grayscale'
INPUT_SHAPE_P1 = (224, 224, 1) # Must match our color mode (1 channel)
LEARNING_RATE = 0.0001
EPOCHS = 30

# --- 2. Load Data ---
print("Loading Path 1 data...")
train_gen, val_gen = create_data_generators(
    dataset_path=DATASET_PATH,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode=COLOR_MODE_P1
)

# --- 3. Build Model ---
print("Building Path 1 model...")
model_p1 = build_custom_cnn(input_shape=INPUT_SHAPE_P1)
model_p1.summary() # Print a summary of the model

# --- 4. Compile Model ---
print("Compiling model...")
model_p1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 5. Define Callbacks ---
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# --- 6. Train Model ---
print("Starting training...")
history_p1 = model_p1.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr]
)

print("Training complete for Path 1.")