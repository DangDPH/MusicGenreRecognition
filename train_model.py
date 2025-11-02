# train_model_v2.py
import os
import numpy as np
import cv2
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(" Libraries imported successfully.")

# --- Constants ---
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
           'metal', 'pop', 'reggae', 'rock']

DATA_DIR = "spectrogram_segments"
IMG_SIZE = 224  # MobileNetV2 default
BATCH_SIZE = 32
BASE_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
LEARNING_RATE = 1e-4

# ============================================================
# Step 1: Load and Preprocess Spectrograms
# ============================================================
def get_data(data_dir, labels, img_size):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)

        if not os.path.exists(path):
            print(f"[Warning] Missing folder: {path}")
            continue

        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # BGR→RGB
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"️ Could not load image {img}: {e}")

    return np.array(data, dtype=object)

def preprocess_data(train_data, val_data, img_size):
    x_train, y_train, x_val, y_val = [], [], [], []

    for feature, label in train_data:
        x_train.append(feature)
        y_train.append(label)
    for feature, label in val_data:
        x_val.append(feature)
        y_val.append(label)

    x_train = np.array(x_train) / 255.0
    x_val = np.array(x_val) / 255.0

    x_train = x_train.reshape(-1, img_size, img_size, 3)
    x_val = x_val.reshape(-1, img_size, img_size, 3)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    return x_train, y_train, x_val, y_val

# ============================================================
# Step 2: Build the MobileNetV2 Transfer Model
# ============================================================
def build_transfer_model(input_shape, num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # Freeze for base training

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    return model

# ============================================================
# Step 3: Train the Model (Base + Fine-tuning)
# ============================================================
def plot_history(history, filename="model_history.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Acc")
    plt.plot(epochs_range, val_acc, label="Val Acc")
    plt.legend(); plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(); plt.title("Loss")

    plt.savefig(filename)
    plt.close()
    print(f" Saved accuracy/loss plot to {filename}")

# ============================================================
# Step 4: Evaluate the Model
# ============================================================
def evaluate_model(model, x_val, y_val, labels, filename="confusion_matrix.png"):
    preds = np.argmax(model.predict(x_val), axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_val, preds, target_names=labels))

    cm = confusion_matrix(y_val, preds)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(10, 8))
    sn.heatmap(df_cm, annot=True, fmt='g', cmap='Purples')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(filename)
    plt.close()
    print(f" Saved confusion matrix to {filename}")

# ============================================================
# Main Execution
# ============================================================
def main():
    print("\n--- Step 1: Loading and Preprocessing Spectrogram Data ---")
    train_data = get_data(os.path.join(DATA_DIR, "train"), LABELS, IMG_SIZE)
    val_data = get_data(os.path.join(DATA_DIR, "test"), LABELS, IMG_SIZE)

    if len(train_data) == 0 or len(val_data) == 0:
        print("[ERROR] No data found. Run preprocessing first.")
        return

    x_train, y_train, x_val, y_val = preprocess_data(train_data, val_data, IMG_SIZE)
    print(f"Training samples: {x_train.shape[0]}, Validation samples: {x_val.shape[0]}")

    # --- Data Augmentation ---
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)

    # --- Step 2: Build Model ---
    print("\n--- Step 2: Building MobileNetV2 Transfer Model ---")
    model = build_transfer_model((IMG_SIZE, IMG_SIZE, 3), len(LABELS))
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # --- Step 3: Train Model ---
    print("\n--- Step 3: Training Model (Base Training) ---")
    base_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("mobilenetv2_base.h5", save_best_only=True)
    ]
    history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        epochs=BASE_EPOCHS,
                        validation_data=(x_val, y_val),
                        callbacks=base_callbacks)

    # Fine-tuning
    print("\n--- Step 3: Fine-tuning Deeper Layers ---")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    fine_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("mobilenetv2_finetuned.h5", save_best_only=True)
    ]

    history_fine = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                             epochs=FINE_TUNE_EPOCHS,
                             validation_data=(x_val, y_val),
                             callbacks=fine_callbacks)

    for k in history.history.keys():
        history.history[k] += history_fine.history[k]

    print("\n Model fine-tuning complete.")
    model.save("mobilenetv2_final_model.h5")
    print(" Model saved as mobilenetv2_final_model.h5")

    # --- Step 4: Evaluate Model ---
    plot_history(history)
    evaluate_model(model, x_val, y_val, LABELS)

if __name__ == "__main__":
    main()
