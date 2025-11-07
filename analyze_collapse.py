#!/usr/bin/env python3
"""
Analisi embedding collapse attraverso le epoche.

Analizza:
1. Distribuzione norme degli embeddings (se normalizzati, dovrebbero essere 1.0)
2. Effetto del margin nelle diverse fasi
3. Distanza media dal centroide globale
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Path alla run da analizzare
RUN_DIR = Path("/Users/arjuna/Progetti/siamese/runs/run_20251106_232135")

def load_data():
    """Carica train_history e batch_metrics"""
    history = pd.read_csv(RUN_DIR / "logs" / "train_history.csv")
    batch_train = pd.read_csv(RUN_DIR / "logs" / "batch_metrics_train.csv")
    batch_val = pd.read_csv(RUN_DIR / "logs" / "batch_metrics_val.csv")

    return history, batch_train, batch_val


def analyze_margin_effect(history):
    """
    Analizza l'effetto del margin attraverso le fasi.

    Nel CurriculumContrastiveLoss:
    - Warmup (epoch 1-5):     margin = margin_init (1.0), weight = 0.8
    - Transition (epoch 6-10): margin = margin_init * 0.75 (0.75), weight = 1.0
    - Hard (epoch 11-20):     margin = margin_init * 0.5 (0.5), weight = 1.2
    """

    print("\n" + "="*80)
    print("2. ANALISI EFFETTO MARGIN")
    print("="*80)

    # Aggiungi colonna margin basata su mining_strategy
    def get_margin(row):
        strategy = row['mining_strategy']
        if strategy == 'random':
            return 1.0  # warmup
        elif strategy == 'semi-hard':
            return 0.75  # transition
        else:  # hard
            return 0.5  # hard

    history['margin'] = history.apply(get_margin, axis=1)

    # Calcola "margin effectiveness" = gap / margin
    # Se il margin è efficace, il gap dovrebbe essere proporzionale al margin
    history['gap_train'] = history['train_d_inter_mean'] - history['train_d_intra_mean']
    history['gap_val'] = history['val_d_inter_mean'] - history['val_d_intra_mean']
    history['margin_effectiveness_train'] = history['gap_train'] / history['margin']
    history['margin_effectiveness_val'] = history['gap_val'] / history['margin']

    # Analisi per fase
    phases = [
        ('Random (1-5)', history[history['epoch'] <= 5]),
        ('Semi-Hard (6-10)', history[(history['epoch'] > 5) & (history['epoch'] <= 10)]),
        ('Hard (11-20)', history[history['epoch'] > 10])
    ]

    print("\nMargin Effectiveness per Fase:")
    print("-" * 80)
    print(f"{'Phase':<20} {'Margin':<10} {'Gap (Train)':<15} {'Gap (Val)':<15} {'Effectiveness':<15}")
    print("-" * 80)

    for phase_name, phase_data in phases:
        margin = phase_data['margin'].iloc[0]
        gap_train = phase_data['gap_train'].mean()
        gap_val = phase_data['gap_val'].mean()
        effectiveness = phase_data['margin_effectiveness_train'].mean()

        print(f"{phase_name:<20} {margin:<10.2f} {gap_train:<15.3f} {gap_val:<15.3f} {effectiveness:<15.3f}")

    print("\n⚠️  Se 'Effectiveness' diminuisce, il margin non è più efficace!")
    print("    Un margin più basso dovrebbe comunque mantenere un buon gap relativo.\n")

    # Calcola "compression rate" = variazione gap rispetto a fase precedente
    print("\nCompression Rate tra Fasi:")
    print("-" * 80)

    gaps = []
    for phase_name, phase_data in phases:
        gap_mean = phase_data['gap_train'].mean()
        gaps.append((phase_name, gap_mean))

    for i in range(1, len(gaps)):
        prev_phase, prev_gap = gaps[i-1]
        curr_phase, curr_gap = gaps[i]
        compression = ((prev_gap - curr_gap) / prev_gap) * 100
        print(f"{prev_phase} → {curr_phase}: Gap ridotto del {compression:.1f}%")

    return history


def analyze_centroid_distance(history):
    """
    Analizza la distanza media dal centroide globale.

    Se gli embeddings collassano, la distanza media dai centroidi
    dei cluster (sia intra che inter) diminuisce.

    Possiamo stimare la "dispersione" come:
    - Se d_intra e d_inter diminuiscono insieme → collapse
    - Se d_intra diminuisce ma d_inter resta stabile → buono
    """

    print("\n" + "="*80)
    print("3. ANALISI DISTANZA DAL CENTROIDE GLOBALE")
    print("="*80)

    # Calcola "average distance" = media di d_intra e d_inter (stima dello spazio utilizzato)
    history['avg_dist_train'] = (history['train_d_intra_mean'] + history['train_d_inter_mean']) / 2
    history['avg_dist_val'] = (history['val_d_intra_mean'] + history['val_d_inter_mean']) / 2

    # Calcola "space utilization" rispetto alla prima epoca
    space_init_train = history['avg_dist_train'].iloc[0]
    space_init_val = history['avg_dist_val'].iloc[0]

    history['space_utilization_train'] = (history['avg_dist_train'] / space_init_train) * 100
    history['space_utilization_val'] = (history['avg_dist_val'] / space_init_val) * 100

    print("\nSpace Utilization (rispetto all'epoch 1):")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Mining':<12} {'Avg Dist (Train)':<18} {'Avg Dist (Val)':<18} {'Space % (Train)':<18} {'Space % (Val)':<15}")
    print("-" * 80)

    for _, row in history.iterrows():
        epoch = int(row['epoch'])
        strategy = row['mining_strategy']
        avg_train = row['avg_dist_train']
        avg_val = row['avg_dist_val']
        space_train = row['space_utilization_train']
        space_val = row['space_utilization_val']

        # Colora in base alla riduzione di spazio
        if space_train < 50:
            marker = "❌"
        elif space_train < 70:
            marker = "⚠️"
        else:
            marker = "✓"

        print(f"{epoch:<8} {strategy:<12} {avg_train:<18.3f} {avg_val:<18.3f} {space_train:<18.1f}% {space_val:<15.1f}% {marker}")

    print("\n⚠️  Se lo spazio utilizzato scende sotto il 50%, c'è un severo collapse!")
    print("    Gli embeddings stanno collassando verso un punto comune.\n")

    # Calcola il ratio di dispersione: d_inter / d_intra
    # Un ratio alto = buona separazione
    history['dispersion_ratio_train'] = history['train_d_inter_mean'] / history['train_d_intra_mean']
    history['dispersion_ratio_val'] = history['val_d_inter_mean'] / history['val_d_intra_mean']

    print("\nDispersion Ratio (d_inter / d_intra):")
    print("-" * 80)
    print(f"{'Epoch':<8} {'Mining':<12} {'Ratio (Train)':<18} {'Ratio (Val)':<18} {'Status':<15}")
    print("-" * 80)

    for _, row in history.iterrows():
        epoch = int(row['epoch'])
        strategy = row['mining_strategy']
        ratio_train = row['dispersion_ratio_train']
        ratio_val = row['dispersion_ratio_val']

        # Un ratio >3 è buono
        if ratio_train > 4.0:
            status = "✓ Good"
        elif ratio_train > 3.0:
            status = "⚠️  Moderate"
        else:
            status = "❌ Poor"

        print(f"{epoch:<8} {strategy:<12} {ratio_train:<18.2f} {ratio_val:<18.2f} {status:<15}")

    print("\n⚠️  Ratio ideale > 4.0. Se scende sotto 3.0, la separazione è insufficiente!\n")

    return history


def create_visualizations(history):
    """Crea grafici riassuntivi"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = history['epoch']

    # 1. Gap attraverso le epoche
    ax = axes[0, 0]
    ax.plot(epochs, history['gap_train'], 'o-', label='Train Gap', linewidth=2)
    ax.plot(epochs, history['gap_val'], 's-', label='Val Gap', linewidth=2)
    ax.axvline(5.5, color='red', linestyle='--', alpha=0.5, label='Random→Semi-Hard')
    ax.axvline(10.5, color='orange', linestyle='--', alpha=0.5, label='Semi-Hard→Hard')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (d_inter - d_intra)')
    ax.set_title('Gap Analysis: Absolute Separation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Space Utilization
    ax = axes[0, 1]
    ax.plot(epochs, history['space_utilization_train'], 'o-', label='Train Space %', linewidth=2)
    ax.plot(epochs, history['space_utilization_val'], 's-', label='Val Space %', linewidth=2)
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Collapse Threshold')
    ax.axvline(5.5, color='red', linestyle='--', alpha=0.3)
    ax.axvline(10.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Space Utilization (%)')
    ax.set_title('Embedding Space Compression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Margin Effectiveness
    ax = axes[1, 0]
    ax.plot(epochs, history['margin_effectiveness_train'], 'o-', label='Train', linewidth=2)
    ax.plot(epochs, history['margin_effectiveness_val'], 's-', label='Val', linewidth=2)
    ax.plot(epochs, history['margin'], 'x--', label='Margin Value', alpha=0.5)
    ax.axvline(5.5, color='red', linestyle='--', alpha=0.3)
    ax.axvline(10.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap / Margin')
    ax.set_title('Margin Effectiveness (Gap normalized by Margin)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Dispersion Ratio
    ax = axes[1, 1]
    ax.plot(epochs, history['dispersion_ratio_train'], 'o-', label='Train', linewidth=2)
    ax.plot(epochs, history['dispersion_ratio_val'], 's-', label='Val', linewidth=2)
    ax.axhline(3.0, color='orange', linestyle='--', alpha=0.5, label='Moderate Threshold')
    ax.axhline(4.0, color='green', linestyle='--', alpha=0.5, label='Good Threshold')
    ax.axvline(5.5, color='red', linestyle='--', alpha=0.3)
    ax.axvline(10.5, color='orange', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('d_inter / d_intra')
    ax.set_title('Dispersion Ratio (Relative Separation)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = RUN_DIR / "analysis_collapse.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Grafici salvati in: {output_path}\n")

    return fig


def main():
    print("\n" + "="*80)
    print("ANALISI EMBEDDING COLLAPSE")
    print("="*80)
    print(f"Run: {RUN_DIR.name}")
    print("="*80)

    # Carica dati
    history, batch_train, batch_val = load_data()

    # 2. Analizza effetto margin
    history = analyze_margin_effect(history)

    # 3. Analizza distanza da centroide
    history = analyze_centroid_distance(history)

    # Crea visualizzazioni
    create_visualizations(history)

    print("\n" + "="*80)
    print("SOMMARIO CONCLUSIONI")
    print("="*80)

    # Calcola metriche chiave
    gap_initial = history['gap_train'].iloc[0]
    gap_final = history['gap_train'].iloc[-1]
    gap_reduction = ((gap_initial - gap_final) / gap_initial) * 100

    space_final = history['space_utilization_train'].iloc[-1]
    ratio_initial = history['dispersion_ratio_train'].iloc[0]
    ratio_final = history['dispersion_ratio_train'].iloc[-1]

    print(f"\n1. Gap Reduction: {gap_reduction:.1f}% (da {gap_initial:.3f} a {gap_final:.3f})")
    if gap_reduction > 50:
        print("   ❌ CRITICAL: Gap ridotto di oltre 50%! Severe collapse.")
    elif gap_reduction > 30:
        print("   ⚠️  WARNING: Gap ridotto di oltre 30%. Moderate collapse.")
    else:
        print("   ✓ OK: Riduzione controllata del gap.")

    print(f"\n2. Space Utilization: {space_final:.1f}%")
    if space_final < 50:
        print("   ❌ CRITICAL: Embeddings hanno collassato! Meno del 50% dello spazio utilizzato.")
    elif space_final < 70:
        print("   ⚠️  WARNING: Significativa compressione dello spazio.")
    else:
        print("   ✓ OK: Spazio ben utilizzato.")

    print(f"\n3. Dispersion Ratio: {ratio_final:.2f} (variazione: {ratio_initial:.2f} → {ratio_final:.2f})")
    if ratio_final < 3.0:
        print("   ❌ CRITICAL: Ratio troppo basso! Separazione insufficiente.")
    elif ratio_final < 4.0:
        print("   ⚠️  WARNING: Ratio moderato. Potrebbe migliorare.")
    else:
        print("   ✓ OK: Buona separazione relativa.")

    print("\n" + "="*80)
    print("RACCOMANDAZIONI")
    print("="*80)

    if gap_reduction > 30 or space_final < 70:
        print("\n⚠️  Il training mostra segni di embedding collapse!")
        print("\nPossibili soluzioni:")
        print("  1. Aumentare margin_init (es. 1.5 o 2.0)")
        print("  2. Mantenere margin più alto nella fase hard (es. 0.8 invece di 0.5)")
        print("  3. Ridurre weight nella fase hard (da 1.2 a 1.0)")
        print("  4. Aggiungere dispersion loss term")
        print("  5. Limitare hard mining agli ultimi 5 epochs invece di 10")
    else:
        print("\n✓ Il training sembra stabile, ma monitora il gap assoluto.")
        print("\nPossibili miglioramenti:")
        print("  1. Provare margin più alti per aumentare il gap assoluto")
        print("  2. Aggiungere early stopping basato su gap minimo")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
