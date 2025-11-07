import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Carica i dati
print("Caricamento dati...")
df = pd.read_csv('/Users/arjuna/Progetti/siamese/data/starts01.csv')

# Seleziona SOLO le colonne ECG (escludi età, genere, etc.)
ecg_columns = ['first_VentricularRate', 'first_PRInterval', 'first_QRSDuration', 
               'first_QTCorrected', 'first_PAxis', 'first_RAxis', 'first_TAxis', 
               'first_QOnset', 'second_VentricularRate', 'second_PRInterval', 
               'second_QRSDuration', 'second_QTCorrected', 'second_PAxis', 
               'second_RAxis', 'second_TAxis', 'second_QOnset']

print(f"Feature ECG selezionate: {len(ecg_columns)}")
print(f"Features: {ecg_columns}")

# Prepara i dati
X = df[ecg_columns].values

# Gestisci valori mancanti
print(f"\nValori mancanti: {np.isnan(X).sum()}")
X = np.nan_to_num(X, nan=0.0)

# Normalizza i dati
print("Normalizzazione dei dati...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applica t-SNE 3D
print("Applicazione t-SNE (questo potrebbe richiedere alcuni minuti)...")
tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)

print(f"t-SNE completato! Shape: {X_tsne.shape}")

# Crea la visualizzazione 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot senza colorazione
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], 
                    c='steelblue', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_zlabel('t-SNE 3', fontsize=12)
ax.set_title('t-SNE 3D - ECG Features Only (starts01.csv)', fontsize=14, fontweight='bold')

# Salva la figura
output_path = '/Users/arjuna/Progetti/siamese/tsne_3d_ecg_only.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigura salvata in: {output_path}")

# Stampa statistiche
print(f"\nStatistiche:")
print(f"Numero di campioni: {len(df)}")
print(f"Numero di feature ECG: {len(ecg_columns)}")

plt.show()
