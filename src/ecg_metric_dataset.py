import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import List, Dict, Tuple
import random
from tqdm import tqdm


class ECGPairDataset(Dataset):
    """
    Dataset per generare pairs di ECG per Contrastive Loss.
    Supporta diverse strategie di mining (random, semi-hard, batch-hard).
    """

    def __init__(self, csv_path: str, mining_strategy: str = "random",
                 embeddings: np.ndarray = None, margin: float = 2.0):
        """
        Args:
            csv_path: path a train/val/test.csv
            mining_strategy: "random", "semi-hard", "batch-hard"
            embeddings: array (N, D) di embeddings per mining strategies
            margin: margin per contrastive loss
        """
        print(f"Loading CSV from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.mining_strategy = mining_strategy
        self.embeddings = embeddings
        self.margin = margin

        # Features: le 13 features ECG
        self.feature_cols = [
            'VentricularRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
            'PAxis', 'RAxis', 'TAxis', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset'
        ]

        # Normalizza features
        print("Normalizing features...")
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.scaler_mean = self.features.mean(axis=0)
        self.scaler_std = self.features.std(axis=0)
        self.features = (self.features - self.scaler_mean) / (self.scaler_std + 1e-8)

        self.patient_ids = self.df['PatientID'].values
        print(f"Building patient indices for {len(np.unique(self.patient_ids))} unique patients...")
        self.unique_patients = np.unique(self.patient_ids)

        # Costruisci indici per paziente
        self.patient_indices = {}
        for patient_id in tqdm(self.unique_patients, desc="Building patient index", leave=False):
            mask = self.patient_ids == patient_id
            self.patient_indices[patient_id] = np.where(mask)[0]

        print("Dataset loaded successfully!")

    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """
        Genera lista di (anchor_idx, positive_idx, label).
        label=0 per same patient (positive), label=1 per different patient (negative)
        """
        pairs = []

        if self.mining_strategy == "random":
            pairs = self._random_mining()
        elif self.mining_strategy == "semi-hard":
            pairs = self._semi_hard_mining()
        elif self.mining_strategy == "batch-hard":
            pairs = self._batch_hard_mining()

        return pairs

    def _random_mining(self) -> List[Tuple[int, int, int]]:
        """Random sampling di pairs"""
        pairs = []

        # Per ogni sample, genera coppie
        for idx in range(len(self.df)):
            patient = self.patient_ids[idx]

            # Positivo: altro ECG dello stesso paziente
            pos_indices = self.patient_indices[patient]
            if len(pos_indices) > 1:
                pos_idx = np.random.choice([i for i in pos_indices if i != idx])
                pairs.append((idx, pos_idx, 0))

            # Negativo: ECG di paziente diverso
            other_patients = [p for p in self.unique_patients if p != patient]
            if other_patients:
                neg_patient = np.random.choice(other_patients)
                neg_idx = np.random.choice(self.patient_indices[neg_patient])
                pairs.append((idx, neg_idx, 1))

        return pairs

    def _semi_hard_mining(self) -> List[Tuple[int, int, int]]:
        """
        Semi-hard mining: seleziona negatives dove
        d_positive < d_negative < d_positive + margin
        """
        if self.embeddings is None:
            return self._random_mining()

        pairs = []

        for idx in range(len(self.df)):
            patient = self.patient_ids[idx]
            anchor_emb = self.embeddings[idx]

            # Positivo
            pos_indices = self.patient_indices[patient]
            if len(pos_indices) > 1:
                pos_idx = np.random.choice([i for i in pos_indices if i != idx])
                d_pos = np.linalg.norm(self.embeddings[pos_idx] - anchor_emb)

                # Cerca semi-hard negative
                other_patients = [p for p in self.unique_patients if p != patient]
                semi_hard_negatives = []

                for neg_patient in other_patients:
                    for neg_idx in self.patient_indices[neg_patient]:
                        d_neg = np.linalg.norm(self.embeddings[neg_idx] - anchor_emb)
                        if d_pos < d_neg < d_pos + self.margin:
                            semi_hard_negatives.append(neg_idx)

                if semi_hard_negatives:
                    neg_idx = np.random.choice(semi_hard_negatives)
                else:
                    # Fallback a random
                    neg_patient = np.random.choice(other_patients)
                    neg_idx = np.random.choice(self.patient_indices[neg_patient])

                pairs.append((idx, pos_idx, 0))
                pairs.append((idx, neg_idx, 1))

        return pairs

    def _batch_hard_mining(self) -> List[Tuple[int, int, int]]:
        """
        Batch-hard mining: seleziona hardest positive e hardest negative
        """
        if self.embeddings is None:
            return self._random_mining()

        pairs = []

        for idx in range(len(self.df)):
            patient = self.patient_ids[idx]
            anchor_emb = self.embeddings[idx]

            # Hardest positivo
            pos_indices = [i for i in self.patient_indices[patient] if i != idx]
            if pos_indices:
                distances = [np.linalg.norm(self.embeddings[i] - anchor_emb) for i in pos_indices]
                hard_pos_idx = pos_indices[np.argmax(distances)]

                # Hardest negativo
                other_patients = [p for p in self.unique_patients if p != patient]
                hardest_neg_distance = float('inf')
                hardest_neg_idx = None

                for neg_patient in other_patients:
                    for neg_idx in self.patient_indices[neg_patient]:
                        d_neg = np.linalg.norm(self.embeddings[neg_idx] - anchor_emb)
                        if d_neg < hardest_neg_distance:
                            hardest_neg_distance = d_neg
                            hardest_neg_idx = neg_idx

                if hardest_neg_idx is not None:
                    pairs.append((idx, hard_pos_idx, 0))
                    pairs.append((idx, hardest_neg_idx, 1))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        anchor_idx, other_idx, label = self.pairs[idx]

        anchor = torch.from_numpy(self.features[anchor_idx])
        other = torch.from_numpy(self.features[other_idx])

        return {
            'anchor': anchor,
            'other': other,
            'label': torch.tensor(label, dtype=torch.float32),
            'anchor_patient': self.patient_ids[anchor_idx],
            'other_patient': self.patient_ids[other_idx]
        }


class PKSampler(Sampler):
    """
    Sampler che seleziona P pazienti e K ECG per paziente.
    Garantisce batch di P*K samples.
    """

    def __init__(self, patient_ids: np.ndarray, P: int = 64, K: int = 4, shuffle: bool = True):
        """
        Args:
            patient_ids: array di patient IDs
            P: numero di pazienti per batch
            K: numero di ECG per paziente
            shuffle: se True, mischia pazienti e samples
        """
        self.patient_ids = patient_ids
        self.P = P
        self.K = K
        self.shuffle = shuffle

        # Costruisci indici per paziente (veloce)
        print("Building PKSampler index...")
        unique_patients = np.unique(patient_ids)
        self.patient_indices = {}
        for patient_id in tqdm(unique_patients, desc="Patient indices", leave=False):
            mask = patient_ids == patient_id
            self.patient_indices[patient_id] = np.where(mask)[0].tolist()

        self.patients = list(self.patient_indices.keys())
        print(f"PKSampler ready: {len(self.patients)} patients, {P}x{K} batch size")

    def __iter__(self):
        # Copia lista di pazienti per non modificare self.patients
        patients = self.patients.copy()

        # Shuffle una sola volta
        if self.shuffle:
            random.shuffle(patients)

        batch = []
        patient_idx = 0

        # Genera batch continuamente finché ci sono pazienti
        while patient_idx < len(patients):
            # Prendi P pazienti per il batch
            batch_patients = patients[patient_idx:patient_idx + self.P]
            patient_idx += self.P

            batch = []
            for patient_id in batch_patients:
                indices = self.patient_indices[patient_id]

                if len(indices) >= self.K:
                    # Se paziente ha K+ ECG, seleziona K
                    selected = random.sample(indices, self.K) if self.shuffle else indices[:self.K]
                else:
                    # Se paziente ha <K ECG, prendi tutti e ripeti
                    selected = indices * (self.K // len(indices) + 1)
                    selected = selected[:self.K]

                batch.extend(selected)

            # Yielda il batch quando hai raccolto samples da P pazienti
            if batch:
                yield batch

    def __len__(self) -> int:
        # Numero totale di samples
        return len(self.patient_ids)
