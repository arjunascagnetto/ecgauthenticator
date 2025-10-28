import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Carica il file filtrato
df = pd.read_csv('/Users/arjuna/Progetti/siamese/data/ECG/02filter.csv')

print("=" * 80)
print("PREPROCESSING - SPLIT DATASET 80/10/10")
print("=" * 80)

# Features ECG
ecg_features = ['VentricularRate', 'PRInterval', 'QRSDuration', 'QTInterval',
                'QTCorrected', 'PAxis', 'RAxis', 'TAxis', 'QOnset', 'QOffset',
                'POnset', 'POffset', 'TOffset']

print(f"\nDataset: {len(df)} esami, {df['PatientID'].nunique()} pazienti unici")
print(f"Features ECG: {len(ecg_features)}")

# Step 1: Split a livello paziente
patients = df['PatientID'].unique()
print(f"\nSplitting {len(patients)} pazienti in 80/10/10...")

train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=42
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.5,
    random_state=42
)

print(f"  Train patients: {len(train_patients)}")
print(f"  Val patients: {len(val_patients)}")
print(f"  Test patients: {len(test_patients)}")

# Step 2: Crea dataframe per split
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]

print(f"\n  Train esami: {len(train_df)}")
print(f"  Val esami: {len(val_df)}")
print(f"  Test esami: {len(test_df)}")

# Step 3: Normalizza features usando StandardScaler su train
scaler = StandardScaler()
train_df_scaled = train_df.copy()
val_df_scaled = val_df.copy()
test_df_scaled = test_df.copy()

print(f"\nNormalizing features using StandardScaler (fitted on train)...")
train_df_scaled[ecg_features] = scaler.fit_transform(train_df[ecg_features])
val_df_scaled[ecg_features] = scaler.transform(val_df[ecg_features])
test_df_scaled[ecg_features] = scaler.transform(test_df[ecg_features])

# Step 4: Salva i dataset
train_file = '/Users/arjuna/Progetti/siamese/data/ECG/train.csv'
val_file = '/Users/arjuna/Progetti/siamese/data/ECG/val.csv'
test_file = '/Users/arjuna/Progetti/siamese/data/ECG/test.csv'

train_df_scaled.to_csv(train_file, index=False)
val_df_scaled.to_csv(val_file, index=False)
test_df_scaled.to_csv(test_file, index=False)

print(f"✅ Train saved: {train_file}")
print(f"✅ Val saved: {val_file}")
print(f"✅ Test saved: {test_file}")

# Step 5: Salva scaler per future use
import pickle
scaler_file = '/Users/arjuna/Progetti/siamese/models/scaler.pkl'
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✅ Scaler saved: {scaler_file}")

# Step 6: Statistiche normalizzazione
print("\n" + "=" * 80)
print("NORMALIZATION STATISTICS (Train set)")
print("=" * 80)
print(f"\n{'Feature':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 80)
for feat in ecg_features:
    print(f"{feat:<20} {train_df_scaled[feat].mean():<12.4f} {train_df_scaled[feat].std():<12.4f} {train_df_scaled[feat].min():<12.4f} {train_df_scaled[feat].max():<12.4f}")

print("\n" + "=" * 80)
print("✅ PREPROCESSING COMPLETED")
print("=" * 80)
