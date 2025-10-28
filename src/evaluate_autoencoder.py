import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

from dataset import ECGDataset
from autoencoder import Autoencoder

def evaluate(model, test_loader, device, feature_names):
    """Valuta il modello su test set"""
    model.eval()

    all_inputs = []
    all_outputs = []
    all_embeddings = []
    all_patient_ids = []
    all_exam_ids = []

    with torch.no_grad():
        for exams, patient_ids, exam_ids in tqdm(test_loader, desc="Evaluating"):
            exams = exams.to(device)

            # Forward pass
            x_reconstructed, embeddings = model(exams)

            # Accumula
            all_inputs.append(exams.cpu().numpy())
            all_outputs.append(x_reconstructed.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            all_patient_ids.extend(patient_ids.numpy())
            all_exam_ids.extend(exam_ids.numpy())

    # Concatena
    inputs = np.concatenate(all_inputs, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)
    embeddings = np.concatenate(all_embeddings, axis=0)

    return inputs, outputs, embeddings, all_patient_ids, all_exam_ids

def calculate_metrics(inputs, outputs, embeddings, feature_names):
    """Calcola metriche di ricostruzione"""

    # MSE globale
    mse_global = np.mean((inputs - outputs) ** 2)

    # MAE globale
    mae_global = np.mean(np.abs(inputs - outputs))

    # Per-feature metrics
    per_feature_mse = np.mean((inputs - outputs) ** 2, axis=0)
    per_feature_mae = np.mean(np.abs(inputs - outputs), axis=0)

    # Percentuale errore relativo
    # (considerando scale dei dati normalizzati)
    rmse_global = np.sqrt(mse_global)

    return {
        'mse_global': mse_global,
        'mae_global': mae_global,
        'rmse_global': rmse_global,
        'per_feature_mse': per_feature_mse,
        'per_feature_mae': per_feature_mae,
    }

def main():
    print("=" * 80)
    print("EVALUATING AUTOENCODER")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Carica test dataset
    print("\nLoading test dataset...")
    test_dataset = ECGDataset('/Users/arjuna/Progetti/siamese/data/ECG/test.csv')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    feature_names = test_dataset.get_feature_names()
    print(f"  Test: {len(test_dataset)} esami")
    print(f"  Features: {feature_names}")

    # Carica modello
    print("\nLoading model...")
    model = Autoencoder(input_dim=13, latent_dim=16, dropout_rate=0.4).to(device)
    model_path = '/Users/arjuna/Progetti/siamese/models/autoencoder_best.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Model loaded from: {model_path}")

    # Valuta
    print("\nEvaluating...")
    inputs, outputs, embeddings, patient_ids, exam_ids = evaluate(
        model, test_loader, device, feature_names
    )

    # Calcola metriche
    print("\nCalculating metrics...")
    metrics = calculate_metrics(inputs, outputs, embeddings, feature_names)

    # Salva embeddings 16D
    print("\nSaving embeddings...")
    embeddings_path = '/Users/arjuna/Progetti/siamese/results/embeddings_16d.npy'
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"  ✅ Embeddings saved: {embeddings_path}")

    # Salva reconstruction
    reconstruction_path = '/Users/arjuna/Progetti/siamese/results/reconstruction.npy'
    np.save(reconstruction_path, outputs)
    print(f"  ✅ Reconstructions saved: {reconstruction_path}")

    # Crea dataframe risultati
    results_df = pd.DataFrame({
        'PatientID': patient_ids,
        'ExamID': exam_ids,
    })

    for i, feat in enumerate(feature_names):
        results_df[f'{feat}_input'] = inputs[:, i]
        results_df[f'{feat}_output'] = outputs[:, i]
        results_df[f'{feat}_error'] = np.abs(inputs[:, i] - outputs[:, i])

    # Salva
    results_path = '/Users/arjuna/Progetti/siamese/results/evaluation_details.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  ✅ Evaluation details saved: {results_path}")

    # Report metriche
    print("\n" + "=" * 80)
    print("RECONSTRUCTION METRICS")
    print("=" * 80)
    print(f"\nGlobal Metrics:")
    print(f"  MSE:  {metrics['mse_global']:.6f}")
    print(f"  MAE:  {metrics['mae_global']:.6f}")
    print(f"  RMSE: {metrics['rmse_global']:.6f}")

    print(f"\nPer-Feature Metrics:")
    print(f"\n{'Feature':<20} {'MSE':<12} {'MAE':<12}")
    print("-" * 50)
    for i, feat in enumerate(feature_names):
        print(f"{feat:<20} {metrics['per_feature_mse'][i]:<12.6f} {metrics['per_feature_mae'][i]:<12.6f}")

    # Salva metriche
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'RMSE'],
        'Value': [metrics['mse_global'], metrics['mae_global'], metrics['rmse_global']]
    })

    metrics_path = '/Users/arjuna/Progetti/siamese/results/autoencoder_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✅ Metrics saved: {metrics_path}")

    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETED")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - Embeddings 16D: {embeddings_path}")
    print(f"  - Reconstruction: {reconstruction_path}")
    print(f"  - Evaluation details: {results_path}")
    print(f"  - Metrics: {metrics_path}")
    print(f"\nEmbedding shape: {embeddings.shape} (16D per sample)")

if __name__ == '__main__':
    main()
