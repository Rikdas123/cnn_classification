import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# ====================
# PATH CONFIG
# ====================
train_dir = r"C:\Users\User\Downloads\codes\horse-or-human\train"
val_dir   = r"C:\Users\User\Downloads\codes\horse-or-human\validation"
model_save_path = r"C:\Users\User\Downloads\codes\ic_friendly_best.keras"

# ====================
# PARAMETERS
# ====================
img_size = (64, 64)
batch_size = 32
epochs = 40
learning_rate = 1e-3

# ====================
# LOAD DATASETS
# ====================
train_ds = image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Normalize
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds   = val_ds.map(lambda x, y: (x / 255.0, y))

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# ====================
# DATA AUGMENTATION
# ====================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
])

# ====================
# IC-FRIENDLY CNN + BATCH NORM (FOLDABLE)
# ====================
model = models.Sequential([
    layers.Input(shape=img_size + (3,)),
    data_augmentation,

    # Block 1
    layers.Conv2D(32, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2,2)),

    # Block 2
    layers.Conv2D(64, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2,2)),

    # Block 3
    layers.Conv2D(128, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling2D((2,2)),

    # Block 4
    layers.Conv2D(256, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),

    layers.Conv2D(256, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Head
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# ====================
# COMPILE MODEL
# ====================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

# ====================
# SAVE BEST MODEL
# ====================
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ====================
# TRAIN MODEL
# ====================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint]
)

# ====================
# PLOT TRAINING HISTORY
# ====================
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
