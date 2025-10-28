# Autoencoder: Embedding a 64 Dimensioni per Features ECG

## Obiettivo

Comprimere le 13 features ECG in uno spazio latente a 64 dimensioni, preservando l'informazione necessaria per ricostruire accuratamente i dati originali. Gli embeddings ottenuti verranno utilizzati in fasi successive per analisi di similarità paziente-specifica.

## Architettura

### Encoder: 13D → 64D
```
Input (13) → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
           → Linear(64) → BatchNorm → ReLU
           → Output embedding (64D)
```

### Decoder: 64D → 13D
```
Input (64D) → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
            → Linear(13) → Output (ricostruito, 13D)
```

### Proprietà
- **Total Parameters**: ~5,600
- **Activation**: ReLU per hidden layers, Linear per output (reconstruction)
- **Normalization**: BatchNorm dopo ogni hidden layer
- **Regularization**: Dropout (0.2) + L2 weight decay

## Loss Function

**MSE + L2 Regularization**

```
Loss_total = Loss_MSE + λ * ||W||²_2

Loss_MSE = mean((x - x_reconstructed)²)
λ = 1e-4 (weight decay coefficient)
```

### Razionale
- **MSE**: standard per reconstruction continua, penalizza errori grandi
- **L2 Regularization**: previene overfitting, stabilizza embeddings

## Hyperparameters

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| Batch size | 256 | Bilancia memoria e stabilità gradient |
| Learning rate | 1e-3 | Adam default per convergenza stabile |
| Optimizer | Adam | Adattativo, robusto per questo setup |
| Epochs | 100 | Max epochs, early stopping a ~15 |
| Early stopping patience | 15 | Previene overfitting |
| Dropout rate | 0.2 | Standard per regolarizzazione |
| λ_L2 | 1e-4 | Leggero, evita underfitting |

## Dataset Split

- **Train**: 80% dei pazienti (~42.5K) → ~131.5K esami
- **Validation**: 10% dei pazienti (~5.3K) → ~16.4K esami
- **Test**: 10% dei pazienti (~5.3K) → ~16.4K esami

Split effettuato a livello paziente per evitare data leakage (esami dello stesso paziente non in set diversi).

## Normalizzazione

Features normalizzate con **StandardScaler** (fitted su train set):
- Media: 0
- Std Dev: 1
- Applicato identicamente a validation e test

## Esperimento e Risultati

### File Script

1. **`src/preprocessing.py`**: Split e normalizzazione dataset
2. **`src/dataset.py`**: PyTorch Dataset per batching
3. **`src/autoencoder.py`**: Architettura modello
4. **`src/train_autoencoder.py`**: Training loop con early stopping
5. **`src/evaluate_autoencoder.py`**: Valutazione e metriche

### Esecuzione

```bash
# Step 1: Preprocessing
/Users/arjuna/Progetti/siamese/.siamese/bin/python /Users/arjuna/Progetti/siamese/src/preprocessing.py

# Step 2: Training
/Users/arjuna/Progetti/siamese/.siamese/bin/python /Users/arjuna/Progetti/siamese/src/train_autoencoder.py

# Step 3: Evaluation
/Users/arjuna/Progetti/siamese/.siamese/bin/python /Users/arjuna/Progetti/siamese/src/evaluate_autoencoder.py
```

### Output

1. **`models/autoencoder_best.pth`**: Best model (lowest validation loss)
2. **`models/scaler.pkl`**: StandardScaler per future predictions
3. **`results/training_history.json`**: Loss history per epoch
4. **`results/training_config.json`**: Hyperparameters usati
5. **`results/embeddings_64d.npy`**: Embeddings latenti (shape: [n_samples, 64])
6. **`results/reconstruction.npy`**: Dati ricostruiti
7. **`results/evaluation_details.csv`**: Input/output/error per feature
8. **`results/autoencoder_metrics.csv`**: MSE, MAE, RMSE globali

## Metriche Attese

Target per una buona ricostruzione:
- **MSE < 0.05** (considerando dati normalizzati)
- **MAE < 0.15**
- **Per-feature MAE < 0.2** (per la maggior parte delle features)

Questi valori garantiscono che gli embeddings 64D preservano l'informazione essenziale per compiti di identificazione paziente.

## Validazione della Qualità degli Embeddings

1. **Reconstruction Error**: MSE/MAE su test set
2. **Embedding Space Analysis**:
   - Analizzare se il rapporto intra/inter distance è preservato
   - Visualizzare con t-SNE/UMAP per verifica qualitativa

3. **Downstream Task**:
   - Usare embeddings per KNN classification (identificazione paziente)
   - Comparare con features originali

## Prossimi Step

1. Analizzare gli embeddings 64D rispetto al rapporto intra/inter distance
2. Se la separazione è preservata, procedere con rete siamese su embedding space
3. Se la separazione degrada, considerare dimensionalità diversa o architettura modificata

## Riferimenti

- LeCun et al., "Autoencoders" in Deep Learning book
- Goodfellow et al., Deep Learning (2016) - Capitolo su Autoencoders
