# src/preprocess.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]   # repo root
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

def generate_synthetic_eeg(n_channels=8, duration_s=30, fs=256):
    t = np.arange(0, duration_s, 1/fs)
    data = []
    for ch in range(n_channels):
        freqs = [8 + ch, 12 + ch*0.5]  # some alpha/beta components
        sig = sum(np.sin(2*np.pi*f*t) for f in freqs)
        noise = 0.5 * np.random.randn(len(t))
        data.append(sig + noise)
    return np.vstack(data).T, fs  # shape (samples, channels)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def apply_notch(sig, fs, notch_freq=50.0, Q=30.0):
    b, a = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b, a, sig, axis=0)

def resample_signal(data, orig_fs, target_fs):
    if orig_fs == target_fs:
        return data, orig_fs
    num = int(round(data.shape[0] * float(target_fs) / orig_fs))
    data_rs = signal.resample(data, num, axis=0)
    return data_rs, target_fs

def main():
    target_fs = 128
    raw_file = RAW_DIR / "raw_eeg.csv"
    if raw_file.exists():
        print("Loading raw CSV:", raw_file)
        df = pd.read_csv(raw_file)
        data = df.values
        orig_fs = int(df.attrs.get("fs", 256)) if getattr(df, "attrs", None) else 256
    else:
        print("No raw found — generating synthetic EEG")
        data, orig_fs = generate_synthetic_eeg(n_channels=8, duration_s=60, fs=256)
        pd.DataFrame(data, columns=[f"ch{c}" for c in range(data.shape[1])]).to_csv(raw_file, index=False)
        print("Synthetic raw saved to", raw_file)

    # Notch
    data = apply_notch(data, orig_fs, notch_freq=50.0, Q=30)
    # Bandpass
    b, a = butter_bandpass(0.5, 45, orig_fs, order=4)
    data = signal.filtfilt(b, a, data, axis=0)
    # Resample
    data, fs = resample_signal(data, orig_fs, target_fs)

    # Save processed continuous
    proc_file = PROC_DIR / "processed_eeg.csv"
    pd.DataFrame(data, columns=[f"ch{c}" for c in range(data.shape[1])]).to_csv(proc_file, index=False)
    # Write metadata file
    meta = {"fs": fs}
    with open(PROC_DIR / "meta.txt", "w") as f:
        f.write(str(meta))

    print("Processed saved to", proc_file, "fs=", fs)

if __name__ == "__main__":
    main()
