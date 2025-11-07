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

# Seleziona le colonne numeriche (parametri ECG)
ecg_columns = [col for col in df.columns if col not in 
               ['PatientID', 'FirstExamID', 'FirstExamDate', 'SecondExamID', 
                'SecondExamDate', 'DaysBetween', 'InRange', 'first_Gender', 'second_Gender']]

print(f"Colonne numeriche selezionate: {len(ecg_columns)}")
print(f"Colonne: {ecg_columns}")

# Prepara i dati: combina primo e secondo esame
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

# Colora i punti in base a InRange
colors = ['red' if x else 'blue' for x in df['InRange']]
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], 
                    c=colors, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_zlabel('t-SNE 3', fontsize=12)
ax.set_title('t-SNE 3D - ECG Data (starts01.csv)', fontsize=14, fontweight='bold')

# Legenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', edgecolor='k', label='InRange=True'),
                   Patch(facecolor='blue', edgecolor='k', label='InRange=False')]
ax.legend(handles=legend_elements, loc='upper right')

# Salva la figura
output_path = '/Users/arjuna/Progetti/siamese/tsne_3d_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigura salvata in: {output_path}")

# Mostra alcune statistiche
print(f"\nStatistiche t-SNE:")
print(f"InRange=True: {sum(df['InRange'])} campioni (rosso)")
print(f"InRange=False: {len(df) - sum(df['InRange'])} campioni (blu)")

plt.show()
