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
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import os
import csv
from datetime import datetime

# Setup iniziale - seed per riproducibilità
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

NUM_EPOCHS = 50
SAVE_EMBEDDINGS_EVERY_N_EPOCHS = 5  # Salva embeddings ogni N epoche per visualizzazione 3D


def calculate_map(embeddings, labels, k=10):
    """
    Calcola Mean Average Precision per retrieval.

    Args:
        embeddings: array di shape (N, embedding_dim)
        labels: array di shape (N,) con le classi
        k: numero di nearest neighbors da considerare

    Returns:
        mAP: Mean Average Precision
    """
    n_samples = len(embeddings)
    if n_samples < 2:
        return 0.0

    # Calcola matrice di distanze
    distances = euclidean_distances(embeddings, embeddings)

    # Per ogni sample, calcola AP
    average_precisions = []

    for i in range(n_samples):
        # Ottieni gli indici ordinati per distanza (escludendo se stesso)
        sorted_indices = np.argsort(distances[i])
        sorted_indices = sorted_indices[sorted_indices != i]  # Rimuovi se stesso

        # Limita a k neighbors
        sorted_indices = sorted_indices[:k]

        # Calcola quali sono relevant (stessa classe)
        relevant = labels[sorted_indices] == labels[i]

        if relevant.sum() == 0:
            continue

        # Calcola precision@k per ogni k
        precisions = []
        num_relevant = 0
        for j, is_relevant in enumerate(relevant, 1):
            if is_relevant:
                num_relevant += 1
                precisions.append(num_relevant / j)

        # Average precision per questa query
        if precisions:
            average_precisions.append(np.mean(precisions))

    # Mean Average Precision
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_embedding_metrics(embeddings, labels):
    """
    Calcola tutte le metriche di valutazione degli embeddings.

    Args:
        embeddings: array di shape (N, embedding_dim)
        labels: array di shape (N,) con le classi

    Returns:
        dict con le metriche: db_index, ch_score, map, nmi
    """
    n_samples = len(embeddings)
    n_classes = len(np.unique(labels))

    # Gestione edge cases
    if n_samples < 2 or n_classes < 2:
        return {
            'db_index': 0.0,
            'ch_score': 0.0,
            'map': 0.0,
            'nmi': 0.0
        }

    # Davies-Bouldin Index (più basso è meglio)
    try:
        db_index = davies_bouldin_score(embeddings, labels)
    except:
        db_index = 0.0

    # Calinski-Harabasz Score (più alto è meglio)
    try:
        ch_score = calinski_harabasz_score(embeddings, labels)
    except:
        ch_score = 0.0

    # Mean Average Precision
    map_score = calculate_map(embeddings, labels, k=min(10, n_samples - 1))

    # Normalized Mutual Information
    # Per NMI, usiamo le label vere vs le label vere (sarà sempre 1.0)
    # In realtà, è utile se si fa clustering. Qui calcoliamo una versione
    # basata su k-means per vedere quanto il clustering automatico matcha le classi reali
    from sklearn.cluster import KMeans
    if n_samples >= n_classes:
        try:
            kmeans = KMeans(n_clusters=n_classes, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            nmi = normalized_mutual_info_score(labels, cluster_labels)
        except:
            nmi = 0.0
    else:
        nmi = 0.0

    return {
        'db_index': float(db_index),
        'ch_score': float(ch_score),
        'map': float(map_score),
        'nmi': float(nmi)
    }


class PairDigits(data.Dataset):
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

        # Lista di liste che contiene gli indici degli elementi appartenenti alle singole classi
        self.class_to_indices = [np.where(self.targets == label)[0] for label in range(10)]

        # Genera le coppie
        self.generate_pairs()

    def generate_pairs(self):
        """Genera le coppie, associando ad ogni elemento un nuovo elemento"""
        # Creiamo un vettore di etichette delle coppie
        self.pair_labels = (np.random.rand(len(self.data)) > 0.5).astype(int)

        # paired_idx conterrà i secondi elementi delle coppie
        self.paired_idx = []
        # Scorriamo le etichette delle coppie
        for i, l in enumerate(self.pair_labels):
            # Otteniamo la classe del primo elemento della coppia
            c1 = self.targets[i]
            if l == 0:  # se la coppia è di tipo simile
                # Scegli un elemento della stessa classe
                j = np.random.choice(self.class_to_indices[c1])
            else:  # altrimenti
                # Scegli un elemento di una classe diversa
                c2 = np.random.choice([c for c in range(10) if c != c1])
                j = np.random.choice(self.class_to_indices[c2])
            self.paired_idx.append(j)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Otteniamo la prima e la seconda immagine
        img1 = torch.from_numpy(self.data[i]).unsqueeze(0)  # Aggiungi canale (1, 8, 8)
        img2 = torch.from_numpy(self.data[self.paired_idx[i]]).unsqueeze(0)
        # Otteniamo l'etichetta della coppia
        label = self.pair_labels[i]
        # Label originali delle singole immagini
        label1 = self.targets[i]
        label2 = self.targets[self.paired_idx[i]]

        return img1, img2, label, label1, label2


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        # Encoder condiviso (Phi) - adattato per immagini 8x8
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

    # Genera timestamp per questa run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'Run timestamp: {timestamp}')

    # Crea directory logs e models se non esistono
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Dataset e DataLoader
    train_dataset = PairDigits(train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    # Salva statistiche di normalizzazione del training set per consistenza
    train_data_mean = train_dataset.data.mean()
    train_data_std = train_dataset.data.std()
    print(f'Dataset size: {len(train_dataset)} samples')
    print(f'Image shape: 1x8x8 (channels x height x width)')
    print(f'Training data normalization - Mean: {train_data_mean:.4f}, Std: {train_data_std:.4f}')

    # Creiamo il modello, la loss e l'optimizer
    model = SiameseNetwork(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=2.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Inizializza file CSV per metriche con timestamp
    csv_file = f'logs/train_metrics_{timestamp}.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'loss',
                        'dist_similar_mean', 'dist_similar_min', 'dist_similar_max',
                        'dist_dissimilar_mean', 'dist_dissimilar_min', 'dist_dissimilar_max',
                        'db_index', 'ch_score', 'map', 'nmi'])
    print(f'Logging metrics to {csv_file}')

    # Training loop
    print('Starting training...')
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for batch_idx, (img1, img2, labels, label1, label2) in enumerate(pbar):
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)
            label1, label2 = label1.to(device), label2.to(device)

            optimizer.zero_grad()

            # Forward pass
            embedding1, embedding2 = model(img1, img2)

            # Calcolo della loss
            loss = criterion(embedding1, embedding2, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calcola distanze e metriche
            with torch.no_grad():
                distances = F.pairwise_distance(embedding1, embedding2)

                # Separa distanze per tipo di coppia
                similar_mask = (labels == 0)
                dissimilar_mask = (labels == 1)

                # Distanze coppie simili (l=0)
                if similar_mask.sum() > 0:
                    dist_similar = distances[similar_mask]
                    dist_sim_mean = dist_similar.mean().item()
                    dist_sim_min = dist_similar.min().item()
                    dist_sim_max = dist_similar.max().item()
                else:
                    dist_sim_mean = dist_sim_min = dist_sim_max = 0.0

                # Distanze coppie dissimili (l=1)
                if dissimilar_mask.sum() > 0:
                    dist_dissimilar = distances[dissimilar_mask]
                    dist_dis_mean = dist_dissimilar.mean().item()
                    dist_dis_min = dist_dissimilar.min().item()
                    dist_dis_max = dist_dissimilar.max().item()
                else:
                    dist_dis_mean = dist_dis_min = dist_dis_max = 0.0

                # Calcola metriche di clustering sugli embeddings del batch
                # Usa embedding1 con le sue label originali (label1)
                emb1_np = embedding1.cpu().numpy()
                label1_np = label1.cpu().numpy()

                batch_metrics = compute_embedding_metrics(emb1_np, label1_np)

            # Scrivi metriche nel CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, batch_idx + 1, loss.item(),
                                dist_sim_mean, dist_sim_min, dist_sim_max,
                                dist_dis_mean, dist_dis_min, dist_dis_max,
                                batch_metrics['db_index'], batch_metrics['ch_score'],
                                batch_metrics['map'], batch_metrics['nmi']])

            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dist_sim': f'{dist_sim_mean:.3f}',
                'dist_dis': f'{dist_dis_mean:.3f}',
                'mAP': f'{batch_metrics["map"]:.3f}'
            })

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}')

        # Calcola metriche sull'intero training set ad ogni epoca
        print(f'  Computing metrics on full training set...')
        model.eval()
        train_embeddings_list = []
        train_labels_list = []

        with torch.no_grad():
            for img1, img2, labels, label1, label2 in train_loader:
                img1 = img1.to(device)
                embedding1 = model.forward_one(img1)
                train_embeddings_list.append(embedding1.cpu().numpy())
                train_labels_list.append(label1.numpy())

        train_embeddings_array = np.vstack(train_embeddings_list)
        train_labels_array = np.concatenate(train_labels_list)

        epoch_metrics = compute_embedding_metrics(train_embeddings_array, train_labels_array)
        print(f'  Train Metrics - DB: {epoch_metrics["db_index"]:.4f}, CH: {epoch_metrics["ch_score"]:.2f}, '
              f'mAP: {epoch_metrics["map"]:.4f}, NMI: {epoch_metrics["nmi"]:.4f}')

        model.train()  # Torna in modalità training

        # Salva embeddings ogni N epoche per visualizzazione 3D
        if (epoch + 1) % SAVE_EMBEDDINGS_EVERY_N_EPOCHS == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f'  Saving embeddings for epoch {epoch+1}...')
            model.eval()

            # Usa il dataset digits base per avere le vere label (non le coppie)
            from sklearn.datasets import load_digits
            digits = load_digits()
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                digits.data, digits.target,
                test_size=0.2,
                stratify=digits.target,
                random_state=0
            )

            # Prepara training set con normalizzazione originale
            X_train_reshaped = X_train_raw.reshape(-1, 8, 8).astype(np.float32)
            # USA STESSA NORMALIZZAZIONE DEL TRAINING (già calcolata)
            X_train_normalized = (X_train_reshaped - train_data_mean) / train_data_std

            # Prepara test set con STESSA normalizzazione del training (CONSISTENTE!)
            X_test_reshaped = X_test_raw.reshape(-1, 8, 8).astype(np.float32)
            X_test_normalized = (X_test_reshaped - train_data_mean) / train_data_std  # USA STATS TRAINING!

            # Calcola embeddings TRAINING SET
            train_embeddings_list = []
            with torch.no_grad():
                for i in range(len(X_train_normalized)):
                    img = torch.from_numpy(X_train_normalized[i]).unsqueeze(0).unsqueeze(0).to(device)
                    embedding = model.forward_one(img)
                    train_embeddings_list.append(embedding.cpu().numpy())

            train_embeddings_array = np.vstack(train_embeddings_list)
            train_labels_array = y_train_raw

            # Calcola embeddings TEST SET
            test_embeddings_list = []
            with torch.no_grad():
                for i in range(len(X_test_normalized)):
                    img = torch.from_numpy(X_test_normalized[i]).unsqueeze(0).unsqueeze(0).to(device)
                    embedding = model.forward_one(img)
                    test_embeddings_list.append(embedding.cpu().numpy())

            test_embeddings_array = np.vstack(test_embeddings_list)
            test_labels_array = y_test_raw

            # Calcola metriche su entrambi i set
            train_emb_metrics = compute_embedding_metrics(train_embeddings_array, train_labels_array)
            test_emb_metrics = compute_embedding_metrics(test_embeddings_array, test_labels_array)

            # Salva embeddings TRAINING in file .npz
            train_embedding_file = f'logs/embeddings_train_epoch_{epoch+1:03d}_{timestamp}.npz'
            np.savez(train_embedding_file,
                    embeddings=train_embeddings_array,
                    labels=train_labels_array,
                    epoch=epoch+1,
                    db_index=train_emb_metrics['db_index'],
                    ch_score=train_emb_metrics['ch_score'],
                    map=train_emb_metrics['map'],
                    nmi=train_emb_metrics['nmi'])
            print(f'  Train embeddings saved to {train_embedding_file}')

            # Salva embeddings TEST in file .npz
            test_embedding_file = f'logs/embeddings_test_epoch_{epoch+1:03d}_{timestamp}.npz'
            np.savez(test_embedding_file,
                    embeddings=test_embeddings_array,
                    labels=test_labels_array,
                    epoch=epoch+1,
                    db_index=test_emb_metrics['db_index'],
                    ch_score=test_emb_metrics['ch_score'],
                    map=test_emb_metrics['map'],
                    nmi=test_emb_metrics['nmi'])
            print(f'  Test embeddings saved to {test_embedding_file}')

            # Stampa confronto metriche
            print(f'  {"="*60}')
            print(f'  EMBEDDING QUALITY COMPARISON (Epoch {epoch+1}):')
            print(f'  {"="*60}')
            print(f'  Metric              | Training    | Test        | Gap')
            print(f'  {"-"*60}')
            print(f'  DB Index (↓ better) | {train_emb_metrics["db_index"]:11.4f} | {test_emb_metrics["db_index"]:11.4f} | {abs(train_emb_metrics["db_index"] - test_emb_metrics["db_index"]):11.4f}')
            print(f'  CH Score (↑ better) | {train_emb_metrics["ch_score"]:11.2f} | {test_emb_metrics["ch_score"]:11.2f} | {abs(train_emb_metrics["ch_score"] - test_emb_metrics["ch_score"]):11.2f}')
            print(f'  mAP                 | {train_emb_metrics["map"]:11.4f} | {test_emb_metrics["map"]:11.4f} | {abs(train_emb_metrics["map"] - test_emb_metrics["map"]):11.4f}')
            print(f'  NMI                 | {train_emb_metrics["nmi"]:11.4f} | {test_emb_metrics["nmi"]:11.4f} | {abs(train_emb_metrics["nmi"] - test_emb_metrics["nmi"]):11.4f}')
            print(f'  {"="*60}')

            # Warning per overfitting
            if train_emb_metrics['map'] - test_emb_metrics['map'] > 0.1:
                print(f'  ⚠️  WARNING: Large train/test mAP gap detected! Possible overfitting.')
            if test_emb_metrics['db_index'] > train_emb_metrics['db_index'] * 1.5:
                print(f'  ⚠️  WARNING: Test DB Index much higher than train! Possible overfitting.')

            model.train()  # Torna in modalità training

    print('Training completed!')

    # Valutazione sul test set
    print('\n' + '='*60)
    print('Evaluating on test set...')
    test_dataset = PairDigits(train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    test_loss = 0
    all_distances_similar = []
    all_distances_dissimilar = []
    test_embeddings_list = []
    test_labels_list = []

    with torch.no_grad():
        for img1, img2, labels, label1, label2 in tqdm(test_loader, desc='Test evaluation'):
            img1, img2, labels = img1.to(device), img2.to(device), labels.float().to(device)

            # Forward pass
            embedding1, embedding2 = model(img1, img2)

            # Calcolo della loss
            loss = criterion(embedding1, embedding2, labels)
            test_loss += loss.item()

            # Calcola distanze
            distances = F.pairwise_distance(embedding1, embedding2)

            # Separa per tipo di coppia
            similar_mask = (labels == 0)
            dissimilar_mask = (labels == 1)

            if similar_mask.sum() > 0:
                all_distances_similar.extend(distances[similar_mask].cpu().numpy())

            if dissimilar_mask.sum() > 0:
                all_distances_dissimilar.extend(distances[dissimilar_mask].cpu().numpy())

            # Raccogli embeddings e label per metriche
            test_embeddings_list.append(embedding1.cpu().numpy())
            test_labels_list.append(label1.numpy())

    # Calcola statistiche finali
    avg_test_loss = test_loss / len(test_loader)

    all_distances_similar = np.array(all_distances_similar)
    all_distances_dissimilar = np.array(all_distances_dissimilar)

    # Calcola metriche di clustering sul test set
    test_embeddings_array = np.vstack(test_embeddings_list)
    test_labels_array = np.concatenate(test_labels_list)
    final_test_metrics = compute_embedding_metrics(test_embeddings_array, test_labels_array)

    print('\n' + '='*60)
    print('TEST SET RESULTS')
    print('='*60)
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'\nSimilar pairs (label=0):')
    print(f'  Mean distance: {all_distances_similar.mean():.4f}')
    print(f'  Min distance:  {all_distances_similar.min():.4f}')
    print(f'  Max distance:  {all_distances_similar.max():.4f}')
    print(f'  Std distance:  {all_distances_similar.std():.4f}')
    print(f'\nDissimilar pairs (label=1):')
    print(f'  Mean distance: {all_distances_dissimilar.mean():.4f}')
    print(f'  Min distance:  {all_distances_dissimilar.min():.4f}')
    print(f'  Max distance:  {all_distances_dissimilar.max():.4f}')
    print(f'  Std distance:  {all_distances_dissimilar.std():.4f}')
    print(f'\nEmbedding Quality Metrics:')
    print(f'  Davies-Bouldin Index: {final_test_metrics["db_index"]:.4f} (lower is better)')
    print(f'  Calinski-Harabasz Score: {final_test_metrics["ch_score"]:.2f} (higher is better)')
    print(f'  Mean Average Precision: {final_test_metrics["map"]:.4f}')
    print(f'  Normalized Mutual Information: {final_test_metrics["nmi"]:.4f}')
    print('='*60)

    # Salva risultati test in un file con timestamp
    test_results_file = f'logs/test_results_{timestamp}.txt'
    with open(test_results_file, 'w') as f:
        f.write('='*60 + '\n')
        f.write('TEST SET RESULTS\n')
        f.write('='*60 + '\n')
        f.write(f'Test Loss: {avg_test_loss:.4f}\n')
        f.write(f'\nSimilar pairs (label=0):\n')
        f.write(f'  Mean distance: {all_distances_similar.mean():.4f}\n')
        f.write(f'  Min distance:  {all_distances_similar.min():.4f}\n')
        f.write(f'  Max distance:  {all_distances_similar.max():.4f}\n')
        f.write(f'  Std distance:  {all_distances_similar.std():.4f}\n')
        f.write(f'\nDissimilar pairs (label=1):\n')
        f.write(f'  Mean distance: {all_distances_dissimilar.mean():.4f}\n')
        f.write(f'  Min distance:  {all_distances_dissimilar.min():.4f}\n')
        f.write(f'  Max distance:  {all_distances_dissimilar.max():.4f}\n')
        f.write(f'  Std distance:  {all_distances_dissimilar.std():.4f}\n')
        f.write(f'\nEmbedding Quality Metrics:\n')
        f.write(f'  Davies-Bouldin Index: {final_test_metrics["db_index"]:.4f} (lower is better)\n')
        f.write(f'  Calinski-Harabasz Score: {final_test_metrics["ch_score"]:.2f} (higher is better)\n')
        f.write(f'  Mean Average Precision: {final_test_metrics["map"]:.4f}\n')
        f.write(f'  Normalized Mutual Information: {final_test_metrics["nmi"]:.4f}\n')
        f.write('='*60 + '\n')
    print(f'\nTest results saved to {test_results_file}')

    # Salva il modello con timestamp
    model_file = f'models/siamese_model_small_{timestamp}.pth'
    torch.save(model.state_dict(), model_file)
    print(f'Model saved as {model_file}')
