import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')  # <- your datasets are directly here
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Ensure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load new datasets (directly from data/)
patients = pd.read_csv(os.path.join(DATA_DIR, 'patients.csv'))
eeg_features = pd.read_csv(os.path.join(DATA_DIR, 'eeg_features.csv'))
mri_features = pd.read_csv(os.path.join(DATA_DIR, 'mri_features.csv'))
labels = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
treatment = pd.read_csv(os.path.join(DATA_DIR, 'treatment.csv'))

# Merge datasets on patient_id
merged = patients.merge(eeg_features, on='patient_id', how='left') \
                 .merge(mri_features, on='patient_id', how='left') \
                 .merge(labels, on='patient_id', how='left') \
                 .merge(treatment, on='patient_id', how='left')

# Save merged dataset
merged_file = os.path.join(PROCESSED_DIR, 'merged_new_dataset.csv')
merged.to_csv(merged_file, index=False)
print(f"✅ Merged dataset saved: {merged_file}")

# Optional: Print summary statistics
print("\n--- Dataset Summary ---")
print(f"Total Patients: {merged['patient_id'].nunique()}")
print(f"Columns: {list(merged.columns)}")
print(f"Missing values:\n{merged.isnull().sum()}")
