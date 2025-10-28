import pandas as pd
import numpy as np
from tqdm import tqdm

# Carica il file filtrato
df = pd.read_csv('/Users/arjuna/Progetti/siamese/data/ECG/02filter.csv')

print("=" * 80)
print("ANALISI CV WITHIN/BETWEEN - FEATURES ECG")
print("=" * 80)

# Features ECG
ecg_features = ['VentricularRate', 'PRInterval', 'QRSDuration', 'QTInterval',
                'QTCorrected', 'PAxis', 'RAxis', 'TAxis', 'QOnset', 'QOffset',
                'POnset', 'POffset', 'TOffset']

results = []

for feature in tqdm(ecg_features, desc="Processando features ECG"):
    # CV within-subject (per ogni paziente)
    cv_within_list = []
    for patient_id in tqdm(df['PatientID'].unique(), desc=f"  Feature: {feature}", leave=False):
        patient_data = df[df['PatientID'] == patient_id][feature]
        if len(patient_data) > 1:  # Almeno 2 misure
            mean_val = patient_data.mean()
            std_val = patient_data.std()
            if mean_val != 0:
                cv_within = (std_val / mean_val) * 100
                cv_within_list.append(cv_within)

    # Media del CV within-subject
    mean_cv_within = np.mean(cv_within_list) if cv_within_list else 0

    # CV between-subject (variazione tra pazienti)
    patient_means = df.groupby('PatientID')[feature].mean()
    global_mean = df[feature].mean()
    between_std = patient_means.std()

    if global_mean != 0:
        cv_between = (between_std / global_mean) * 100
    else:
        cv_between = 0

    # Rapporto CV
    if cv_between != 0:
        cv_ratio = mean_cv_within / cv_between
    else:
        cv_ratio = np.inf

    results.append({
        'Feature': feature,
        'CV_within (%)': mean_cv_within,
        'CV_between (%)': cv_between,
        'Ratio (within/between)': cv_ratio
    })

# Crea dataframe risultati
results_df = pd.DataFrame(results)

# Ordina per rapporto decrescente
results_df_sorted = results_df.sort_values('Ratio (within/between)', ascending=False)

print("\nRISULTATI ORDINATI PER RAPPORTO (DECRESCENTE):\n")
print(results_df_sorted.to_string(index=False))

# Statistiche riassuntive
print("\n" + "=" * 80)
print("STATISTICHE RIASSUNTIVE")
print("=" * 80)
print(f"Rapporto medio: {results_df['Ratio (within/between)'].mean():.4f}")
print(f"Rapporto mediano: {results_df['Ratio (within/between)'].median():.4f}")
print(f"Min rapporto: {results_df['Ratio (within/between)'].min():.4f}")
print(f"Max rapporto: {results_df['Ratio (within/between)'].max():.4f}")

# Interpretazione
print("\n" + "=" * 80)
print("INTERPRETAZIONE")
print("=" * 80)
print("Rapporto < 0.5: Variabilità intra-paziente MOLTO PICCOLA rispetto a inter-paziente")
print("Rapporto 0.5-1.0: Variabilità intra-paziente comparabile/minore a inter-paziente")
print("Rapporto > 1.0: Variabilità intra-paziente PREDOMINA su inter-paziente")

# Salva i risultati
output_file = '/Users/arjuna/Progetti/siamese/data/ECG/cv_analysis_results.csv'
results_df_sorted.to_csv(output_file, index=False)
print(f"\n✅ Risultati salvati in: {output_file}")
