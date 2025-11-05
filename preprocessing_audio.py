# preprocessing_audio.py
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import shutil

# ============ CONFIG ============
DATASET_PATH = "genres"
OUTPUT_PATH = "spectrogram_data"
SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
CLIP_DURATION = 3
FULL_DURATION = 30
# ================================

# --- Prepare output directory ---
print("\n--- Preparing Output Directory ---")
if os.path.exists(OUTPUT_PATH):
    print(f" Removing old folder: {OUTPUT_PATH}")
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f" Created new folder: {OUTPUT_PATH}")

# --- Function to split and save spectrograms ---
def create_mel_spectrograms(audio_path, output_dir, genre, file_idx):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        if len(y) == 0:
            print(f" Skipped empty file: {audio_path}")
            return

        # Pad or trim to 30s exactly
        target_len = int(FULL_DURATION * sr)
        y = librosa.util.fix_length(y, size=target_len)

        # 10 clips of 3 seconds each
        samples_per_clip = int(CLIP_DURATION * sr)
        num_segments = target_len // samples_per_clip

        # Create genre folder
        genre_path = os.path.join(output_dir, genre)
        os.makedirs(genre_path, exist_ok=True)

        # Create one image per clip
        for i in range(num_segments):
            start = i * samples_per_clip
            end = start + samples_per_clip
            y_clip = y[start:end]

            # Compute mel-spectrogram
            S = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Normalize to [0, 1]
            S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())

            # Save each as a separate PNG file
            save_name = f"{genre}_{file_idx:03d}_{i}.png"
            save_path = os.path.join(genre_path, save_name)
            plt.imsave(save_path, S_dB_norm, cmap='magma')


    except Exception as e:
        print(f" Skipped file {audio_path}: {e}")

# --- Process entire dataset ---
def main():
    print("\n--- Generating 3-second Mel-Spectrograms ---")
    genres = [g for g in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, g))]
    total_audio = 0
    total_images = 0

    for genre in genres:
        print(f"\n Processing genre: {genre}")
        genre_path = os.path.join(DATASET_PATH, genre)
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

        for idx, file in enumerate(files):
            file_path = os.path.join(genre_path, file)
            create_mel_spectrograms(file_path, OUTPUT_PATH, genre, idx)
            total_audio += 1
            total_images += FULL_DURATION // CLIP_DURATION

    print(f"\n All spectrograms generated successfully for {len(genres)} genres.")
    print(f" Output folder: {os.path.abspath(OUTPUT_PATH)}")
    #print(f" Created {total_images} spectrogram images from {total_audio} audio files.")

if __name__ == "__main__":
    main()
