import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm import tqdm

# Setup iniziale - seed per riproducibilità
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

NUM_EPOCHS = 50


class PairMNIST(data.Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.mnist = MNIST(root=root, train=train, transform=transform, download=download)

        # lista di liste che contiene gli indici degli elementi appartenenti alle singole classi
        # in pratica class_to_indices[5] contiene gli indici degli elementi di classe 5
        self.class_to_indices = [np.where(self.mnist.targets == label)[0] for label in range(10)]

        # genera le coppie
        self.generate_pairs()

    def generate_pairs(self):
        """Genera le coppie, associando ad ogni elemento di MNIST un nuovo elemento"""
        # creiamo un vettore di etichette delle coppie
        self.pair_labels = (np.random.rand(len(self.mnist)) > 0.5).astype(int)

        # paired_idx conterrà i secondi elementi delle coppie
        self.paired_idx = []
        # scorriamo le etichette delle coppie
        for i, l in enumerate(self.pair_labels):
            # otteniamo la classe del primo elemento della coppia
            c1 = self.mnist.targets[i].item()
            if l == 0:  # se la coppia è di tipo simile
                # scegli un elemento della stessa classe
                j = np.random.choice(self.class_to_indices[c1])
            else:  # altrimenti
                # scegli un elemento di una classe diversa
                # in pratica scegliamo una classe c2 diversa da c1
                c2 = np.random.choice([c for c in range(10) if c != c1])
                # quindi scegliamo un elemento appartenente alla classe c2
                j = np.random.choice(self.class_to_indices[c2])
            self.paired_idx.append(j)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, i):
        # otteniamo la prima e la seconda immagine
        img1, _ = self.mnist[i]
        img2, _ = self.mnist[self.paired_idx[i]]
        # otteniamo l'etichetta della coppia
        label = self.pair_labels[i]

        return img1, img2, label


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        # Encoder condiviso (Phi)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward_one(self, x):
        """Passa una singola immagine attraverso l'encoder"""
        return self.encoder(x)

    def forward(self, img1, img2):
        """Forward pass per una coppia di immagini"""
        # Otteniamo gli embedding per entrambe le immagini
        embedding1 = self.forward_one(img1)
        embedding2 = self.forward_one(img2)

        return embedding1, embedding2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, label):
        # Calcoliamo la distanza Euclidea
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)

        # Calcoliamo la loss
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss


if __name__ == '__main__':
    # Configurazione
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Trasformazioni
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Dataset e DataLoader
    train_dataset = PairMNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    # Creiamo il modello, la loss e l'optimizer
    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print('Starting training...')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for img1, img2, labels in pbar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

            optimizer.zero_grad()

            # Forward pass
            embedding1, embedding2 = model(img1, img2)

            # Calcolo della loss
            loss = criterion(embedding1, embedding2, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}')

    print('Training completed!')

    # Salva il modello
    torch.save(model.state_dict(), 'siamese_model.pth')
    print('Model saved as siamese_model.pth')
