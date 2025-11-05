# train_model_inceptionv3_v2.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
# --- ADD THIS IMPORT ---
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil  # <-- ADDED THIS IMPORT

# ================= CONFIG =================
DATA_DIR = "spectrogram_data"
OUTPUT_DIR = "results_inceptionv3_v2"
IMG_SIZE = (150, 150) 
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 30
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS
# ==========================================

# --- NEW: Clean and prepare output directory ---
if os.path.exists(OUTPUT_DIR):
    print(f"\n--- Removing old results folder: {OUTPUT_DIR} ---")
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- Created new, empty results folder: {OUTPUT_DIR} ---")
# ---

# --- Step 1: Data Loading ---
print("\n--- Step 1: Loading and Preparing Data ---")

train_datagen = ImageDataGenerator(
    # rescale=1./255, # <-- REMOVED THIS (THE BUG)
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    validation_split=0.2
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training',
    class_mode='categorical',
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

# --- Step 2: Build InceptionV3 Transfer Learning Model ---
print("\n--- Step 2: Building InceptionV3 Transfer Learning Model ---")

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False  # Freeze for Phase 1

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- Step 3: Phase 1 - Train Top Layers ---
print("\n--- Step 3: Training Only Top Layers (Phase 1) ---")

callbacks_phase1 = [
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_phase1.keras"), monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

history_phase1 = model.fit(
    train_gen,
    epochs=PHASE1_EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks_phase1
)

# --- Step 4: Phase 2 - Fine-Tuning ---
print("\n--- Step 4: Fine-Tuning the Last Layers (Phase 2) ---")

base_model.trainable = True
for layer in base_model.layers[:-80]:  # Unfreeze last 80 layers for fine-tuning
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks_phase2 = [
    ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_model_finetuned.keras"), monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

history_phase2 = model.fit(
    train_gen,
    initial_epoch=history_phase1.epoch[-1], # <-- Start where Phase 1 left off
    epochs=TOTAL_EPOCHS, # <-- Train up to the total
    validation_data=val_gen,
    callbacks=callbacks_phase2
)

# --- Step 5: Evaluation ---
print("\n--- Step 5: Evaluating the Best Fine-Tuned Model ---")

best_model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "best_model_finetuned.keras"))
val_gen.reset()
Y_pred = best_model.predict(val_gen)
y_pred = np.argmax(Y_pred, axis=1)

print("\n--- Classification Report (Fine-Tuned) ---")
print(classification_report(val_gen.classes, y_pred, target_names=val_gen.class_indices.keys()))

# --- FIXED CONFUSION MATRIX ---
cm = confusion_matrix(val_gen.classes, y_pred)
class_names = list(val_gen.class_indices.keys())

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True, # <-- Show numbers
    fmt='g',    # <-- No scientific notation
    cmap='Blues', # <-- Use a clearer colormap
    xticklabels=class_names, # <-- Add genre labels
    yticklabels=class_names  # <-- Add genre labels
)
plt.title("Confusion Matrix - InceptionV3")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), bbox_inches='tight')
plt.close()

# --- Step 6: Training Curves ---
print("\n--- Step 6: Saving Training Curves ---")

# Combine history from both phases
acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
loss = history_phase1.history['loss'] + history_phase2.history['loss']
val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

plt.figure(figsize=(10,5))
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Training vs Validation Accuracy (Full Run)')
plt.axvline(x=PHASE1_EPOCHS-1, color='red', linestyle='--', label='Fine-Tuning Start')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve_full.png"))
plt.close()

plt.figure(figsize=(10,5))
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Training vs Validation Loss (Full Run)')
plt.axvline(x=PHASE1_EPOCHS-1, color='red', linestyle='--', label='Fine-Tuning Start')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve_full.png"))
plt.close()

print(f"\n Training and evaluation completed successfully. Results saved in: {OUTPUT_DIR}")