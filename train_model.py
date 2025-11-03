# train_model_v2.py
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ============ CONFIG ============
DATA_DIR = "spectrogram_data"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
OUTPUT_DIR = "results"   # Folder to save confusion matrix & graphs
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Step 1: Data Loading ---
print("\n--- Step 1: Loading and Preparing Data ---")

# Create one generator for TRAINING (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,      # Augmentation
    zoom_range=0.1,         # Augmentation
    horizontal_flip=True,   # Augmentation
    validation_split=0.2    # Use 20% of data for validation
)

# Create a SECOND generator for VALIDATION (NO augmentation)
validation_datagen = ImageDataGenerator(
    rescale=1./255,         # Only rescale
    validation_split=0.2    # Must match the training split
)

# ---
train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical'
)

val_gen = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical',
    shuffle=False             # <-- THIS IS THE CRITICAL FIX
)

# --- Step 2: Build Manual VGG16 Model ---
print("\n--- Step 2: Building Manual VGG16-like CNN ---")

def build_cnn_model(input_shape=(128, 128, 3), num_classes=10):
    model = models.Sequential([

        # --- Block 1 ---
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        # --- Block 2 ---
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        # --- Block 3 ---
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        # --- Block 4 ---
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),

        # --- Classification Head ---
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_cnn_model(input_shape=(*IMG_SIZE, 3), num_classes=train_gen.num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- Step 3: Training ---
print("\n--- Step 3: Training the Model ---")
callbacks = [
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_manual_vgg16.keras"), save_best_only=True, verbose=1),
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    EarlyStopping(patience=6, restore_best_weights=True, verbose=1)
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks
)

# --- Step 4: Evaluation ---
print("\n--- Step 4: Evaluating the Best Model ---")
best_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "best_model_manual_vgg16.keras"))

val_gen.reset()
Y_pred = best_model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)

print("\n--- Classification Report ---")
print(classification_report(val_gen.classes, y_pred, target_names=val_gen.class_indices.keys()))

# Confusion Matrix
cm = confusion_matrix(val_gen.classes, y_pred)

# Get the list of genre names from the generator
class_names = list(val_gen.class_indices.keys())

plt.figure(figsize=(12, 10))  # Make the figure larger
sns.heatmap(
    cm,
    annot=True,            # Show the numbers in each cell
    fmt='g',               # Use normal number format (no scientific notation)
    cmap='Blues',          # Use a clearer color map
    xticklabels=class_names, # Add genre names to x-axis
    yticklabels=class_names  # Add genre names to y-axis
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches='tight')
plt.close()

# --- Step 5: Training Graphs ---
print("\n--- Step 5: Saving Training Curves ---")
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
acc_path = os.path.join(OUTPUT_DIR, "accuracy_curve.png")
plt.savefig(acc_path, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
loss_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(loss_path, bbox_inches='tight')
plt.close()

print(f" Accuracy and loss curves saved to {OUTPUT_DIR}")
print("\n Training and evaluation completed successfully.")
