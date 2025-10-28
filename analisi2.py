import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Carica il file filtrato
df = pd.read_csv('/Users/arjuna/Progetti/siamese/data/ECG/02filter.csv')

print("=" * 80)
print("ANALISI INTRA/INTER DISTANZA - MANHATTAN DISTANCE")
print("=" * 80)

# Features ECG
ecg_features = ['VentricularRate', 'PRInterval', 'QRSDuration', 'QTInterval',
                'QTCorrected', 'PAxis', 'RAxis', 'TAxis', 'QOnset', 'QOffset',
                'POnset', 'POffset', 'TOffset']

# Estrai dati ECG
ecg_data = df[ecg_features].values
patient_ids = df['PatientID'].values

print(f"\nDataset: {len(df)} esami, {len(df['PatientID'].unique())} pazienti unici")
print(f"Features: {len(ecg_features)}")

# Mappa patient_id -> indices degli esami
patient_indices = {}
for idx, pid in enumerate(patient_ids):
    if pid not in patient_indices:
        patient_indices[pid] = []
    patient_indices[pid].append(idx)

patients_list = list(patient_indices.keys())
n_patients = len(patients_list)

print(f"\nCalcolo distanze Manhattan...")

# Calcola distanze intra-paziente (within)
intra_distances = []
print("\n1. Calcolando distanze INTRA-paziente...")
for patient_id in tqdm(patients_list, desc="Pazienti (intra)"):
    indices = patient_indices[patient_id]
    if len(indices) > 1:
        patient_exams = ecg_data[indices]
        # Distanza pairwise tra tutti gli esami dello stesso paziente
        distances = cdist(patient_exams, patient_exams, metric='cityblock')
        # Prendi solo triangolo superiore (evita duplicati e diagonale)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        intra_distances.extend(upper_tri)

mean_intra_distance = np.mean(intra_distances) if intra_distances else 0
std_intra_distance = np.std(intra_distances) if intra_distances else 0

print(f"  Distanze intra-paziente calcolate: {len(intra_distances)}")
print(f"  Media: {mean_intra_distance:.4f}")
print(f"  Std Dev: {std_intra_distance:.4f}")

# Calcola distanze inter-paziente (between)
inter_distances = []
print("\n2. Calcolando distanze INTER-paziente...")
for i, patient_id_1 in enumerate(tqdm(patients_list[:-1], desc="Pazienti (inter)")):
    indices_1 = patient_indices[patient_id_1]
    patient_exams_1 = ecg_data[indices_1]

    # Campiona il resto dei pazienti (per efficienza con dataset grande)
    # Prendi un numero limitato di pazienti diversi
    sample_size = min(100, len(patients_list) - i - 1)
    if sample_size > 0:
        sample_indices = np.random.choice(range(i + 1, len(patients_list)),
                                         size=sample_size, replace=False)
        for j in sample_indices:
            patient_id_2 = patients_list[j]
            indices_2 = patient_indices[patient_id_2]
            patient_exams_2 = ecg_data[indices_2]

            # Distanza tra tutti gli esami dei due pazienti
            distances = cdist(patient_exams_1, patient_exams_2, metric='cityblock')
            inter_distances.extend(distances.flatten())

mean_inter_distance = np.mean(inter_distances) if inter_distances else 0
std_inter_distance = np.std(inter_distances) if inter_distances else 0

print(f"  Distanze inter-paziente calcolate: {len(inter_distances)}")
print(f"  Media: {mean_inter_distance:.4f}")
print(f"  Std Dev: {std_inter_distance:.4f}")

# Calcola rapporto
if mean_inter_distance != 0:
    distance_ratio = mean_intra_distance / mean_inter_distance
else:
    distance_ratio = np.inf

print("\n" + "=" * 80)
print("RISULTATI")
print("=" * 80)
print(f"Distanza media INTRA-paziente:  {mean_intra_distance:.4f} ± {std_intra_distance:.4f}")
print(f"Distanza media INTER-paziente:  {mean_inter_distance:.4f} ± {std_inter_distance:.4f}")
print(f"Rapporto (intra/inter):         {distance_ratio:.4f}")

print("\n" + "=" * 80)
print("INTERPRETAZIONE")
print("=" * 80)
if distance_ratio < 0.5:
    print("✅ Rapporto < 0.5: Gli esami dello stesso paziente sono MOLTO PIÙ SIMILI")
    print("   tra loro rispetto agli esami di altri pazienti.")
    print("   → Gli esami del paziente sono una FIRMA DISTINTIVA")
elif distance_ratio < 1.0:
    print("✅ Rapporto 0.5-1.0: Gli esami dello stesso paziente sono COMPARABILMENTE")
    print("   o LEGGERMENTE PIÙ SIMILI tra loro rispetto agli esami di altri pazienti.")
elif distance_ratio < 2.0:
    print("⚠️  Rapporto 1.0-2.0: Gli esami dello stesso paziente sono simili quanto")
    print("   agli esami di altri pazienti. Alta variabilità intra-paziente.")
else:
    print("❌ Rapporto > 2.0: Gli esami dello stesso paziente sono MENO SIMILI")
    print("   tra loro che agli esami di altri pazienti.")
    print("   → PROBLEMA: forte sovrapposizione tra pazienti")

# Salva i risultati
results_summary = {
    'Metrica': ['Distanza media INTRA', 'Distanza media INTER', 'Rapporto (intra/inter)',
                'Std INTRA', 'Std INTER', 'N distanze INTRA', 'N distanze INTER'],
    'Valore': [mean_intra_distance, mean_inter_distance, distance_ratio,
               std_intra_distance, std_inter_distance, len(intra_distances), len(inter_distances)]
}

results_df = pd.DataFrame(results_summary)

output_file = '/Users/arjuna/Progetti/siamese/data/ECG/manhattan_distance_analysis.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✅ Risultati salvati in: {output_file}")

# Salva distribuzione distanze per analisi ulteriore
distances_summary = pd.DataFrame({
    'distance_type': ['intra'] * len(intra_distances) + ['inter'] * len(inter_distances),
    'distance': intra_distances + inter_distances
})

distances_file = '/Users/arjuna/Progetti/siamese/data/ECG/manhattan_distances_distribution.csv'
distances_summary.to_csv(distances_file, index=False)
print(f"✅ Distribuzione distanze salvata in: {distances_file}")
