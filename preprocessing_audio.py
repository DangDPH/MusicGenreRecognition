# 1_preprocess_audio.py
import os
import shutil
import math
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings

print("Libraries for audio processing imported.")

# --- Constants ---
LABELS = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
SOURCE_DIR = "genres"
SAVE_DIR = "spectrogram_segments"

# Audio settings
SAMPLE_RATE = 22050
DURATION = 30 # Full audio file duration
SEGMENT_DURATION = 3 # Duration of each split segment
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

def process_audio_files(source_dir, save_dir):
    """
    Loads all audio files, splits them into 3-second segments,
    creates a spectrogram for each segment, and saves them to
    train/test directories.
    """
    print(f"\n--- Starting Audio Processing ---")
    
    # Clean up old directory
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Removed old directory: {save_dir}")
        
    # Create new directories
    train_path = os.path.join(save_dir, "train")
    test_path = os.path.join(save_dir, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for label in LABELS:
        print(f"Processing genre: {label}")
        
        # Create genre subfolders in train/test
        os.makedirs(os.path.join(train_path, label), exist_ok=True)
        os.makedirs(os.path.join(test_path, label), exist_ok=True)
        
        genre_path = os.path.join(source_dir, label)
        if not os.path.exists(genre_path):
            print(f"  [Error] Directory not found: {genre_path}")
            continue
            
        # Get all audio files for the genre
        audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav') or f.endswith('.mp3')]
        
        # Split files into train (60%) and test (40%) sets
        train_files, test_files = train_test_split(audio_files, test_size=0.4, random_state=42)
        
        # Process training files
        print(f"  Processing {len(train_files)} files for TRAINING...")
        for f in train_files:
            file_path = os.path.join(genre_path, f)
            save_segments_to_images(file_path, os.path.join(train_path, label), label)

        # Process testing files
        print(f"  Processing {len(test_files)} files for TESTING...")
        for f in test_files:
            file_path = os.path.join(genre_path, f)
            save_segments_to_images(file_path, os.path.join(test_path, label), label)

    print("\n--- Audio processing complete. All spectrograms saved. ---")


def save_segments_to_images(audio_path, save_dir, base_filename):
    """
    Loads one audio file, splits it, and saves spectrograms for each segment.
    """
    # Suppress warnings for librosa loading (e.g., for mp3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
        except Exception as e:
            print(f"    [Error] Could not load {audio_path}: {e}")
            return

    num_segments = math.floor(len(signal) / SAMPLES_PER_SEGMENT)
    
    for s in range(num_segments):
        start_sample = s * SAMPLES_PER_SEGMENT
        end_sample = start_sample + SAMPLES_PER_SEGMENT
        segment = signal[start_sample:end_sample]
        
        # Ensure segment is long enough
        if len(segment) == SAMPLES_PER_SEGMENT:
            try:
                X = librosa.stft(segment)
                Xdb = librosa.amplitude_to_db(abs(X))
                
                plt.figure(figsize=(14, 5))
                librosa.display.specshow(Xdb, sr=sr)
                
                segment_filename = f"{base_filename}_{os.path.basename(audio_path)}_{s+1}.png"
                plt.savefig(os.path.join(save_dir, segment_filename))
                plt.close()
                
            except Exception as e:
                print(f"    [Error] Could not process segment {s+1} of {audio_path}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    process_audio_files(source_dir=SOURCE_DIR, save_dir=SAVE_DIR)