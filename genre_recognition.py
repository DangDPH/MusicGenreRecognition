import os
import sys
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.cm as cm
import matplotlib 
from scipy.stats import mode
import cv2
import warnings

# ============ CONFIG ============
# --- This MUST match your trained InceptionV3 model ---
MODEL_PATH = "results_inceptionv3_v2/best_model_finetuned.keras" 
GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

# --- These MUST match your preprocessing_audio.py ---
IMG_SIZE = (150, 150) # InceptionV3 input size
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
CLIP_DURATION = 3  # seconds
SAMPLES_PER_CLIP = int(CLIP_DURATION * SAMPLE_RATE)
# ================================


def create_spectrogram_batch(audio_path):
    """
    Loads one audio file, processes it just like the training data,
    and returns a batch of spectrogram segments ready for the model.
    """
    
    # Suppress warnings for librosa loading (e.g., for mp3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            # Pad or trim to 30s to be consistent
            target_len = int(30 * sr)
            y = librosa.util.fix_length(y, size=target_len)
        except Exception as e:
            print(f"  [Error] Could not load {audio_path}: {e}")
            return None

    # 2. Split into 3-second segments and create spectrograms
    num_segments = len(y) // SAMPLES_PER_CLIP
    segments_batch = []
    
    # Get the 'magma' colormap object (Fixed Deprecation Warning)
    cmap = matplotlib.colormaps.get_cmap('magma') # <-- FIXED

    for i in range(num_segments):
        start = i * SAMPLES_PER_CLIP
        end = start + SAMPLES_PER_CLIP
        y_clip = y[start:end]

        if len(y_clip) == SAMPLES_PER_CLIP:
            S = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # --- Apply the EXACT same normalization as preprocessing_audio.py ---
            S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
            
            # --- Apply the 'magma' colormap (converts to 3 channels) ---
            # This converts [0, 1] grayscale to [0, 1] RGB
            img_rgb = cmap(S_dB_norm)[..., :3]
            
            # --- Convert to [0, 255] range (like a saved PNG) ---
            img_rgb_255 = (img_rgb * 255).astype(np.uint8)
            
            # --- Resize to the model's input size ---
            img_resized = cv2.resize(img_rgb_255, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
            
            segments_batch.append(img_resized)

    if not segments_batch:
        return None
    
    # Convert batch to numpy array
    image_batch = np.array(segments_batch)
    
    # --- Apply the CORRECT InceptionV3 preprocessing ---
    # This scales from [0, 255] to [-1, 1]
    preprocessed_batch = preprocess_input(image_batch.astype(np.float32))
    
    return preprocessed_batch


def predict_genre(model, file_path):
    """
    Predicts the genre of a single audio file.
    """
    print(f"\nProcessing: {file_path}")
    
    # 1. Preprocess the song into a batch of segments
    spectrograms = create_spectrogram_batch(file_path)
    
    if spectrograms is None or spectrograms.shape[0] == 0:
        print("  ...could not process this file.")
        return

    # 2. Predict on all 10 segments at once
    preds = model.predict(spectrograms)
    
    # 3. Get the winning class index for each segment
    predicted_indices = np.argmax(preds, axis=1) # This is a list of NUMBERS

    # 4. Timeline predictions
    print("\n--- Genre Prediction Timeline (every 3 seconds) ---")
    time_labels = [f"{(i*CLIP_DURATION):02d}-{(i+1)*CLIP_DURATION:02d}s" for i in range(len(preds))]
    
    for i, pred_vector in enumerate(preds):
        top_idx = predicted_indices[i] # Get the index
        top_genre = GENRES[top_idx]    # Convert index to string
        top_conf = pred_vector[top_idx] * 100
        print(f"  Segment {time_labels[i]}: {top_genre} ({top_conf:5.2f}%)")

    # 5. Overall Prediction (Majority Vote)
    # --- FIX ---
    # Call mode() on the NUMERIC array 'predicted_indices'
    final_index = mode(predicted_indices, keepdims=False)[0] 
    final_genre = GENRES[final_index] # Convert the final index to a string
    
    # Calculate vote count from the numeric list
    vote_count = np.count_nonzero(predicted_indices == final_index)
    
    print("\n--- Final Estimated Genre (Majority Vote) ---")
    print(f"==> Predicted Genre: {final_genre.upper()}")
    print(f"    (Won {vote_count} out of {len(predicted_indices)} segments)")


def main():
    # --- 1. Check for command-line argument ---
    if len(sys.argv) < 2:
        print("Error: No audio file provided.")
        print("Usage: python genre_recognition.py \"path/to/your/song.wav\"")
        sys.exit(1)
        
    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    # --- 2. Load Model ---
    print(f"--- Loading Model from {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        print(f"[Fatal Error] Model file not found at {MODEL_PATH}")
        print("Please run train_model.py first to generate the model.")
        sys.exit(1)
        
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # --- 3. Predict ---
    predict_genre(model, file_path)


if __name__ == "__main__":
    # Suppress TensorFlow logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore', category=UserWarning)
    main()

