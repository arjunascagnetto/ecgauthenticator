import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup iniziale - seed per riproducibilità
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

NUM_EPOCHS = 50


class TripletDigits(data.Dataset):
    def __init__(self, train=True, train_split=0.8):
        # Carica il dataset digits da scikit-learn
        digits = load_digits()

        # Separa train e test con stratificazione per mantenere proporzioni delle classi
        X_train, X_test, y_train, y_test = train_test_split(
            digits.data, digits.target,
            test_size=(1 - train_split),
            stratify=digits.target,
            random_state=0
        )

        if train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test

        # Reshape da (64,) a (8, 8) e normalizza
        self.data = self.data.reshape(-1, 8, 8).astype(np.float32)
        # Normalizzazione (i valori sono 0-16)
        self.data = (self.data - self.data.mean()) / self.data.std()

        # Indici per classe
        self.class_to_indices = [np.where(self.targets == label)[0] for label in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Anchor
        anchor_img = self.data[idx]
        anchor_label = self.targets[idx]

        # Positive: stessa classe dell'anchor
        positive_idx = np.random.choice(self.class_to_indices[anchor_label])
        positive_img = self.data[positive_idx]

        # Negative: classe diversa dall'anchor
        negative_label = np.random.choice([c for c in range(10) if c != anchor_label])
        negative_idx = np.random.choice(self.class_to_indices[negative_label])
        negative_img = self.data[negative_idx]

        # Converti in tensori e aggiungi canale
        anchor_img = torch.from_numpy(anchor_img).unsqueeze(0)  # (1, 8, 8)
        positive_img = torch.from_numpy(positive_img).unsqueeze(0)
        negative_img = torch.from_numpy(negative_img).unsqueeze(0)

        return anchor_img, positive_img, negative_img


class TripletNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TripletNetwork, self).__init__()

        # Encoder condiviso - adattato per immagini 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4 -> 2x2
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_one(self, x):
        return self.encoder(x)

    def forward(self, anchor, positive, negative):
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)

        return anchor_emb, positive_emb, negative_emb


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Distanza anchor-positive
        distance_positive = F.pairwise_distance(anchor, positive)
        # Distanza anchor-negative
        distance_negative = F.pairwise_distance(anchor, negative)

        # Triplet loss
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


if __name__ == '__main__':
    # Configurazione
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Dataset e DataLoader
    triplet_dataset = TripletDigits(train=True)
    triplet_loader = DataLoader(triplet_dataset, batch_size=64, shuffle=True, num_workers=0)

    print(f'Dataset size: {len(triplet_dataset)} samples')
    print(f'Image shape: 1x8x8 (channels x height x width)')

    # Creiamo il modello, la loss e l'optimizer
    model = TripletNetwork(embedding_dim=128).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print('Starting training...')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(triplet_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            # Forward pass
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)

            # Calcolo della loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(triplet_loader)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}')

    print('Training completed!')

    # Salva il modello
    torch.save(model.state_dict(), 'triplet_model_small.pth')
    print('Model saved as triplet_model_small.pth')
