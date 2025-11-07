import numpy as np
import torch
from typing import List, Tuple, Optional


class MiningStrategy:
    """Base class per mining strategies"""

    def __init__(self, embeddings: Optional[np.ndarray] = None, margin: float = 2.0):
        self.embeddings = embeddings
        self.margin = margin

    def get_pairs(self, batch_patient_ids: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Ritorna lista di (anchor_idx, other_idx, label) per il batch.
        label=0 per positivi, label=1 per negativi
        """
        raise NotImplementedError


class RandomMining(MiningStrategy):
    """Random mining: seleziona pairs casuali"""

    def get_pairs(self, batch_patient_ids: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Args:
            batch_patient_ids: array di patient IDs nel batch (lunghezza = batch_size)
        Returns:
            pairs: lista di (anchor_idx, other_idx, label)
        """
        pairs = []
        batch_size = len(batch_patient_ids)

        for anchor_idx in range(batch_size):
            anchor_patient = batch_patient_ids[anchor_idx]

            # Seleziona positivo casuale (stesso paziente)
            positive_indices = np.where(batch_patient_ids == anchor_patient)[0]
            positive_indices = positive_indices[positive_indices != anchor_idx]

            if len(positive_indices) > 0:
                pos_idx = np.random.choice(positive_indices)
                pairs.append((anchor_idx, pos_idx, 0))

            # Seleziona negativo casuale (paziente diverso)
            negative_indices = np.where(batch_patient_ids != anchor_patient)[0]
            if len(negative_indices) > 0:
                neg_idx = np.random.choice(negative_indices)
                pairs.append((anchor_idx, neg_idx, 1))

        return pairs


class SemiHardMining(MiningStrategy):
    """
    Semi-hard mining: seleziona negatives dove
    d_positive < d_negative < d_positive + margin
    """

    def get_pairs(self, batch_patient_ids: np.ndarray,
                  batch_embeddings: torch.Tensor) -> List[Tuple[int, int, int]]:
        """
        Args:
            batch_patient_ids: patient IDs nel batch
            batch_embeddings: embeddings del batch (batch_size, embedding_dim)
        Returns:
            pairs: lista di (anchor_idx, other_idx, label)
        """
        if batch_embeddings is None:
            return RandomMining().get_pairs(batch_patient_ids)

        pairs = []
        batch_size = len(batch_patient_ids)
        embeddings = batch_embeddings.detach().cpu().numpy()

        for anchor_idx in range(batch_size):
            anchor_patient = batch_patient_ids[anchor_idx]
            anchor_emb = embeddings[anchor_idx]

            # Positivo casuale
            positive_indices = np.where(batch_patient_ids == anchor_patient)[0]
            positive_indices = positive_indices[positive_indices != anchor_idx]

            if len(positive_indices) > 0:
                pos_idx = np.random.choice(positive_indices)
                d_pos = np.linalg.norm(embeddings[pos_idx] - anchor_emb)

                # Cerca semi-hard negatives
                negative_indices = np.where(batch_patient_ids != anchor_patient)[0]
                semi_hard_negatives = []

                for neg_idx in negative_indices:
                    d_neg = np.linalg.norm(embeddings[neg_idx] - anchor_emb)
                    if d_pos < d_neg < d_pos + self.margin:
                        semi_hard_negatives.append(neg_idx)

                if semi_hard_negatives:
                    neg_idx = np.random.choice(semi_hard_negatives)
                else:
                    # Fallback: negativo casuale
                    neg_idx = np.random.choice(negative_indices) if len(negative_indices) > 0 else None

                pairs.append((anchor_idx, pos_idx, 0))
                if neg_idx is not None:
                    pairs.append((anchor_idx, neg_idx, 1))

        return pairs


class BatchHardMining(MiningStrategy):
    """
    Batch-hard mining: seleziona hardest positive e hardest negative per ogni anchor
    """

    def get_pairs(self, batch_patient_ids: np.ndarray,
                  batch_embeddings: torch.Tensor) -> List[Tuple[int, int, int]]:
        """
        Args:
            batch_patient_ids: patient IDs nel batch
            batch_embeddings: embeddings del batch
        Returns:
            pairs: lista di (anchor_idx, other_idx, label)
        """
        if batch_embeddings is None:
            return RandomMining().get_pairs(batch_patient_ids)

        pairs = []
        batch_size = len(batch_patient_ids)
        embeddings = batch_embeddings.detach().cpu().numpy()

        for anchor_idx in range(batch_size):
            anchor_patient = batch_patient_ids[anchor_idx]
            anchor_emb = embeddings[anchor_idx]

            # Hardest positivo (massima distanza da anchor)
            positive_indices = np.where(batch_patient_ids == anchor_patient)[0]
            positive_indices = positive_indices[positive_indices != anchor_idx]

            if len(positive_indices) > 0:
                distances = np.array([np.linalg.norm(embeddings[i] - anchor_emb)
                                    for i in positive_indices])
                hard_pos_idx = positive_indices[np.argmax(distances)]

                # Hardest negativo (minima distanza da anchor, ma > 0)
                negative_indices = np.where(batch_patient_ids != anchor_patient)[0]
                if len(negative_indices) > 0:
                    distances = np.array([np.linalg.norm(embeddings[i] - anchor_emb)
                                        for i in negative_indices])
                    hard_neg_idx = negative_indices[np.argmin(distances)]

                    pairs.append((anchor_idx, hard_pos_idx, 0))
                    pairs.append((anchor_idx, hard_neg_idx, 1))

        return pairs


class AdaptiveMining(MiningStrategy):
    """
    Adaptive mining: seleziona la strategia in base alle metriche di clustering.
    - Se metriche buone: usa batch-hard
    - Se metriche degradate: usa semi-hard o random
    """

    def __init__(self, embeddings: Optional[np.ndarray] = None, margin: float = 2.0):
        super().__init__(embeddings, margin)
        self.current_strategy = "random"
        self.ch_threshold_good = 15.0
        self.ch_threshold_bad = 5.0
        self.db_threshold_good = 2.5
        self.db_threshold_bad = 4.0

    def update_strategy(self, clustering_metrics: dict):
        """
        Aggiorna la strategia in base alle metriche di clustering.

        Args:
            clustering_metrics: dict con 'calinski_harabasz', 'davies_bouldin', ecc.
        """
        ch = clustering_metrics.get('calinski_harabasz', 0)
        db = clustering_metrics.get('davies_bouldin', float('inf'))

        # Fallback a random se collapse rilevato
        if ch < self.ch_threshold_bad or db > self.db_threshold_bad:
            self.current_strategy = "random"
        # Semi-hard se metriche moderate
        elif ch < self.ch_threshold_good or db > self.db_threshold_good:
            self.current_strategy = "semi-hard"
        # Batch-hard se metriche buone
        else:
            self.current_strategy = "batch-hard"

    def get_pairs(self, batch_patient_ids: np.ndarray,
                  batch_embeddings: Optional[torch.Tensor] = None) -> List[Tuple[int, int, int]]:
        """
        Applica la strategia corrente.
        """
        if self.current_strategy == "random":
            return RandomMining().get_pairs(batch_patient_ids)
        elif self.current_strategy == "semi-hard":
            return SemiHardMining(margin=self.margin).get_pairs(
                batch_patient_ids, batch_embeddings)
        elif self.current_strategy == "batch-hard":
            return BatchHardMining(margin=self.margin).get_pairs(
                batch_patient_ids, batch_embeddings)
