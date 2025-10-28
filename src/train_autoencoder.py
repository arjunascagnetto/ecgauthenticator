import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

from dataset import ECGDataset
from autoencoder import Autoencoder, AutoencoderLoss

# Configuration
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'input_dim': 13,
    'latent_dim': 16,
    'dropout_rate': 0.4,
    'batch_size': 256,
    'num_epochs': 100,
    'learning_rate': 5e-4,
    'lambda_l2': 1e-4,
    'early_stopping_patience': 5,
}

def train_epoch(model, criterion, optimizer, train_loader, device):
    """Addestra per un'epoca"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_l2 = 0

    for batch_idx, (exams, _, _) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        exams = exams.to(device)

        # Forward pass
        x_reconstructed, _ = model(exams)

        # Calcola loss
        loss_total, loss_mse, loss_l2 = criterion(exams, x_reconstructed)

        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumula losses
        total_loss += loss_total.item()
        total_mse += loss_mse.item()
        total_l2 += loss_l2.item()

    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'l2': total_l2 / n_batches,
    }

def validate(model, criterion, val_loader, device):
    """Valida il modello"""
    model.eval()
    total_loss = 0
    total_mse = 0

    with torch.no_grad():
        for exams, _, _ in tqdm(val_loader, desc="Validating", leave=False):
            exams = exams.to(device)

            # Forward pass
            x_reconstructed, _ = model(exams)

            # Calcola loss (solo MSE per validation)
            loss_total, loss_mse, _ = criterion(exams, x_reconstructed)

            total_loss += loss_total.item()
            total_mse += loss_mse.item()

    n_batches = len(val_loader)
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
    }

def main():
    print("=" * 80)
    print("TRAINING AUTOENCODER")
    print("=" * 80)

    device = CONFIG['device']
    print(f"\nDevice: {device}")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    # Carica dataset
    print("\nLoading datasets...")
    train_dataset = ECGDataset('/Users/arjuna/Progetti/siamese/data/ECG/train.csv')
    val_dataset = ECGDataset('/Users/arjuna/Progetti/siamese/data/ECG/val.csv')

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)} esami")
    print(f"  Val: {len(val_dataset)} esami")

    # Crea modello
    print("\nCreating model...")
    model = Autoencoder(
        input_dim=CONFIG['input_dim'],
        latent_dim=CONFIG['latent_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)

    print(f"  Model:\n{model}")

    # Loss e optimizer
    criterion = AutoencoderLoss(model, lambda_l2=CONFIG['lambda_l2'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Training loop con early stopping
    print("\nTraining...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_metrics = train_epoch(model, criterion, optimizer, train_loader, device)

        # Validate
        val_metrics = validate(model, criterion, val_loader, device)

        # Log
        log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_mse': train_metrics['mse'],
            'train_l2': train_metrics['l2'],
            'val_loss': val_metrics['loss'],
            'val_mse': val_metrics['mse'],
        }
        history.append(log)

        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | "
              f"Train Loss: {train_metrics['loss']:.6f} (MSE: {train_metrics['mse']:.6f}, L2: {train_metrics['l2']:.6f}) | "
              f"Val Loss: {val_metrics['loss']:.6f}")

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Salva best model
            model_path = '/Users/arjuna/Progetti/siamese/models/autoencoder_best.pth'
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Best model saved (val_loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                break

    # Salva history
    history_path = '/Users/arjuna/Progetti/siamese/results/training_history.json'
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training history saved: {history_path}")

    # Salva config
    config_path = '/Users/arjuna/Progetti/siamese/results/training_config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✅ Config saved: {config_path}")

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    main()
