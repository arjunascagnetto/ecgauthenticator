import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss standard.
    loss = (1-Y) * D^2 / 2 + Y * max(margin - D, 0)^2 / 2
    where D = ||x1 - x2||_2
    """

    def __init__(self, margin: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding1: (N, D) embeddings
            embedding2: (N, D) embeddings
            label: (N,) binary labels (0=positivo, 1=negativo)
        Returns:
            loss: scalar
        """
        cosine_similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        cosine_distance = 1 - cosine_similarity

        loss = (1 - label) * torch.pow(cosine_distance, 2) / 2 + \
               label * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2) / 2

        return torch.mean(loss)


class AdaptiveContrastiveLoss(nn.Module):
    """
    Contrastive Loss con margin adattivo basato sulla fase di training.

    - Warmup (epoch 1-10): margin alto (es. 2.0) per incoraggiare separazione
    - Transition (epoch 11-25): margin moderato (1.5)
    - Hard mining (epoch 26+): margin basso (1.0) per fine-tuning
    """

    def __init__(self, margin_init: float = 2.0, margin_final: float = 1.0,
                 total_epochs: int = 50):
        super(AdaptiveContrastiveLoss, self).__init__()
        self.margin_init = margin_init
        self.margin_final = margin_final
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_margin = margin_init

    def update_epoch(self, epoch: int):
        """Aggiorna il margin in base all'epoca"""
        self.current_epoch = epoch
        # Decay lineare del margin
        progress = epoch / max(self.total_epochs, 1)
        self.current_margin = self.margin_init - \
                             (self.margin_init - self.margin_final) * progress

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding1: (N, D) embeddings
            embedding2: (N, D) embeddings
            label: (N,) binary labels (0=positivo, 1=negativo)
        Returns:
            loss: scalar
        """
        cosine_similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        cosine_distance = 1 - cosine_similarity

        loss = (1 - label) * torch.pow(cosine_distance, 2) / 2 + \
               label * torch.pow(torch.clamp(self.current_margin - cosine_distance, min=0.0), 2) / 2

        return torch.mean(loss)


class CurriculumContrastiveLoss(nn.Module):
    """
    Contrastive Loss con curriculum learning.
    Supporta fasi progressive configurabili (warmup, transition, hard).

    Fasi:
    - Epoch 1 to random_epochs: Warmup (weight=0.8, margin=margin_init)
    - Epoch (random_epochs+1) to semihard_epochs: Transition (weight=1.0, margin=margin_init*0.75)
    - Epoch (semihard_epochs+1) onwards: Hard (weight=1.2, margin=margin_init*0.5)
    """

    def __init__(self, margin_init: float = 2.0, total_epochs: int = 50,
                 random_epochs: int = 10, semihard_epochs: int = 25):
        """
        Args:
            margin_init: margin iniziale
            total_epochs: numero totale di epoche
            random_epochs: fine della fase warmup
            semihard_epochs: fine della fase transition
        """
        super(CurriculumContrastiveLoss, self).__init__()
        self.margin_init = margin_init
        self.total_epochs = total_epochs
        self.random_epochs = random_epochs
        self.semihard_epochs = semihard_epochs
        self.current_epoch = 0
        self.current_phase = "warmup"  # warmup, transition, hard

    def update_epoch(self, epoch: int):
        """Aggiorna fase in base all'epoca e parametri configurati"""
        self.current_epoch = epoch

        if epoch <= self.random_epochs:
            self.current_phase = "warmup"
        elif epoch <= self.semihard_epochs:
            self.current_phase = "transition"
        else:
            self.current_phase = "hard"

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding1: (N, D) embeddings
            embedding2: (N, D) embeddings
            label: (N,) binary labels (0=positivo, 1=negativo)
        Returns:
            loss: scalar
        """
        cosine_similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        cosine_distance = 1 - cosine_similarity

        # Margin decresce con le epoche
        if self.current_phase == "warmup":
            margin = self.margin_init
            # Peso ridotto per far passare meno segnale
            weight = 0.8
        elif self.current_phase == "transition":
            margin = self.margin_init * 0.75
            weight = 1.0
        else:  # hard
            margin = self.margin_init * 0.5
            weight = 1.2  # Peso aumentato per forzare fine-tuning

        loss = (1 - label) * torch.pow(cosine_distance, 2) / 2 + \
               label * torch.pow(torch.clamp(margin - cosine_distance, min=0.0), 2) / 2

        return torch.mean(loss) * weight


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss (Wang et al. 2019).
    Combina hard positives, hard negatives e easy negatives in un'unica loss.
    Più stabile rispetto a Contrastive Loss, riduce collapse.

    loss = 1/N * sum_i [ log(1 + sum_j exp(-α(d_ij_pos - λ))) +
                         log(1 + sum_k exp(β(d_ik_neg - λ))) ]
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0, lambda_param: float = 0.5):
        """
        Args:
            alpha: peso per positive similarities
            beta: peso per negative similarities
            lambda_param: threshold di similarità
        """
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_param = lambda_param

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (N, D) embeddings
            labels: (N,) patient IDs per identificare positivi/negativi
        Returns:
            loss: scalar
        """
        # Calcola pairwise similarities
        similarities = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        # Loss per ogni anchor
        loss = 0.0
        batch_size = embeddings.size(0)

        for i in range(batch_size):
            # Positivi: stesso paziente (escludendo se stesso)
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            pos_sim = similarities[i][pos_mask]

            # Negativi: paziente diverso
            neg_mask = labels != labels[i]
            neg_sim = similarities[i][neg_mask]

            if len(pos_sim) > 0:
                # Positive term: log(1 + sum exp(-α(d_pos - λ)))
                pos_loss = torch.log(1 + torch.sum(torch.exp(-self.alpha * (pos_sim - self.lambda_param))))
                loss += pos_loss

            if len(neg_sim) > 0:
                # Negative term: log(1 + sum exp(β(d_neg - λ)))
                neg_loss = torch.log(1 + torch.sum(torch.exp(self.beta * (neg_sim - self.lambda_param))))
                loss += neg_loss

        return loss / batch_size
