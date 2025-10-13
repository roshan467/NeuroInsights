# src/feature_extraction.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import os

# === Paths ===
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# === EEG Bands ===
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}

# === EEG Feature Extraction Functions ===
def bandpower_from_psd(freqs, psd, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    return np.trapz(psd[idx], freqs[idx])

def extract_eeg_features(data, fs, epoch_sec=2):
    n_samples, n_ch = data.shape
    epoch_len = int(epoch_sec * fs)
    features = []
    for start in range(0, n_samples - epoch_len + 1, epoch_len):
        epoch = data[start:start+epoch_len, :]
        bandpowers = []
        for ch in range(n_ch):
            freqs, psd = welch(epoch[:, ch], fs=fs, nperseg=epoch_len//2)
            ch_band = [bandpower_from_psd(freqs, psd, fmin, fmax) for fmin,fmax in BANDS.values()]
            bandpowers.append(ch_band)
        bandpowers = np.array(bandpowers)  # (n_ch, n_bands)
        mean_bp = bandpowers.mean(axis=0)
        features.append(mean_bp)
    return np.array(features)  # shape (n_epochs, n_bands)

# === Main Function ===
def main():
    # 1️⃣ Extract EEG features from processed EEG
    proc_file = PROC_DIR / "processed_eeg.csv"
    if not proc_file.exists():
        raise FileNotFoundError(proc_file, "Run preprocess first.")
    df = pd.read_csv(proc_file)
    fs = 128
    data = df.values
    eeg_feats = extract_eeg_features(data, fs=fs, epoch_sec=2)
    eeg_col_names = [f"{band}_power" for band in BANDS.keys()]
    eeg_features_df = pd.DataFrame(eeg_feats, columns=eeg_col_names)

    # 2️⃣ Load merged dataset (patients + EEG + MRI + labels + treatment)
    merged_file = PROC_DIR / "merged_new_dataset.csv"
    if not merged_file.exists():
        raise FileNotFoundError(merged_file, "Run preprocess merge_new_datasets first.")
    merged = pd.read_csv(merged_file)

    # Example: merge EEG features (mean over epochs) with patient-level data
    eeg_mean_feats = pd.DataFrame(eeg_features_df.mean(axis=0)).T
    eeg_mean_feats['patient_id'] = merged['patient_id'].iloc[0]  # example mapping; adjust as needed

    # Combine EEG features with other features
    final_features = merged.merge(eeg_mean_feats, on='patient_id', how='left')

    # Save final features
    out_file = FEATURE_DIR / "final_features.csv"
    final_features.to_csv(out_file, index=False)
    print("✅ Saved final features to", out_file)

if __name__ == "__main__":
    main()

