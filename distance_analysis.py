import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

# Carica i dati
print("Caricamento dati...")
df = pd.read_csv('/Users/arjuna/Progetti/siamese/data/starts01.csv')

# Seleziona SOLO le colonne ECG
ecg_columns = ['first_VentricularRate', 'first_PRInterval', 'first_QRSDuration', 
               'first_QTCorrected', 'first_PAxis', 'first_RAxis', 'first_TAxis', 
               'first_QOnset', 'second_VentricularRate', 'second_PRInterval', 
               'second_QRSDuration', 'second_QTCorrected', 'second_PAxis', 
               'second_RAxis', 'second_TAxis', 'second_QOnset']

# Dividi in primo e secondo esame
first_exam_cols = [col for col in ecg_columns if col.startswith('first_')]
second_exam_cols = [col for col in ecg_columns if col.startswith('second_')]

# Normalizza i dati
print("Normalizzazione dei dati...")
scaler = StandardScaler()
X_all = df[ecg_columns].values
X_all_scaled = scaler.fit_transform(X_all)

# Dividi in primo e secondo esame (dopo normalizzazione)
X_first = X_all_scaled[:, :len(first_exam_cols)]
X_second = X_all_scaled[:, len(first_exam_cols):]

print(f"\nDati caricati:")
print(f"Pazienti: {len(df)}")
print(f"Feature per esame: {len(first_exam_cols)}")

# ==================== DISTANZE INTRA-PAZIENTE ====================
print("\n" + "="*60)
print("DISTANZE INTRA-PAZIENTE (tra primo e secondo esame)")
print("="*60)

intra_distances = []
for i in range(len(df)):
    dist = euclidean(X_first[i], X_second[i])
    intra_distances.append(dist)

intra_distances = np.array(intra_distances)

print(f"\nMedia distanze intra-paziente: {np.mean(intra_distances):.6f}")
print(f"Std distanze intra-paziente:   {np.std(intra_distances):.6f}")
print(f"Min distanza intra-paziente:   {np.min(intra_distances):.6f}")
print(f"Max distanza intra-paziente:   {np.max(intra_distances):.6f}")
print(f"Mediana distanze intra-paziente: {np.median(intra_distances):.6f}")

# ==================== DISTANZE INTER-PAZIENTE ====================
print("\n" + "="*60)
print("DISTANZE INTER-PAZIENTE (tra pazienti diversi)")
print("="*60)

# Calcola le distanze tra TUTTI i pazienti (usando primo esame come rappresentante)
inter_distances = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        dist = euclidean(X_first[i], X_first[j])
        inter_distances.append(dist)

inter_distances = np.array(inter_distances)

print(f"\nMedia distanze inter-paziente: {np.mean(inter_distances):.6f}")
print(f"Std distanze inter-paziente:   {np.std(inter_distances):.6f}")
print(f"Min distanza inter-paziente:   {np.min(inter_distances):.6f}")
print(f"Max distanza inter-paziente:   {np.max(inter_distances):.6f}")
print(f"Mediana distanze inter-paziente: {np.median(inter_distances):.6f}")

# ==================== CONFRONTO ====================
print("\n" + "="*60)
print("CONFRONTO E STATISTICHE")
print("="*60)

print(f"\nRapporto (media inter / media intra): {np.mean(inter_distances) / np.mean(intra_distances):.4f}")
print(f"Separabilità: gli esami dello stesso paziente sono {np.mean(inter_distances) / np.mean(intra_distances):.2f}x più lontani")
print(f"           rispetto a esami diversi dello stesso paziente")

# Analizza minime e massime
print(f"\nDistanza minima intra-paziente:  {np.min(intra_distances):.6f} (std: {np.std(np.min(intra_distances)):.6f})")
print(f"Distanza massima intra-paziente: {np.max(intra_distances):.6f} (std: {np.std(np.max(intra_distances)):.6f})")
print(f"Distanza minima inter-paziente:  {np.min(inter_distances):.6f}")
print(f"Distanza massima inter-paziente: {np.max(inter_distances):.6f}")

# Percentili
print(f"\nPercentili distanze intra-paziente:")
for p in [25, 50, 75, 90, 95]:
    print(f"  {p}° percentile: {np.percentile(intra_distances, p):.6f}")

print(f"\nPercentili distanze inter-paziente:")
for p in [25, 50, 75, 90, 95]:
    print(f"  {p}° percentile: {np.percentile(inter_distances, p):.6f}")

# Distribuzioni
print(f"\nDistribuzioni:")
print(f"Intra-paziente - Q1: {np.percentile(intra_distances, 25):.6f}, Q3: {np.percentile(intra_distances, 75):.6f}, IQR: {np.percentile(intra_distances, 75) - np.percentile(intra_distances, 25):.6f}")
print(f"Inter-paziente - Q1: {np.percentile(inter_distances, 25):.6f}, Q3: {np.percentile(inter_distances, 75):.6f}, IQR: {np.percentile(inter_distances, 75) - np.percentile(inter_distances, 25):.6f}")

# Salva i risultati
results = {
    'intra_mean': np.mean(intra_distances),
    'intra_std': np.std(intra_distances),
    'intra_min': np.min(intra_distances),
    'intra_max': np.max(intra_distances),
    'inter_mean': np.mean(inter_distances),
    'inter_std': np.std(inter_distances),
    'inter_min': np.min(inter_distances),
    'inter_max': np.max(inter_distances),
}

print(f"\n" + "="*60)
print("SUMMARY (formato compatto per visualizzazione)")
print("="*60)
print(f"INTRA:  μ={results['intra_mean']:.6f} σ={results['intra_std']:.6f} [min={results['intra_min']:.6f}, max={results['intra_max']:.6f}]")
print(f"INTER:  μ={results['inter_mean']:.6f} σ={results['inter_std']:.6f} [min={results['inter_min']:.6f}, max={results['inter_max']:.6f}]")

