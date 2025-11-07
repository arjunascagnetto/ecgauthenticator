import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ECGEncoder(nn.Module):
    """
    Flexible MLP encoder per ECG con architettura configurabile.

    Supporta:
    - Numero variabile di hidden layers
    - Dropout su ogni layer
    - BatchNorm su ogni layer
    - L2 normalization output opzionale

    Architettura:
    input_dim → hidden_layers[0] → hidden_layers[1] → ... → embedding_dim
    Ogni layer: Linear → BatchNorm → ELU → Dropout
    """

    def __init__(self, input_dim: int = 13, hidden_dims: List[int] = None,
                 embedding_dim: int = 32, dropout: float = 0.2, normalize: bool = True):
        """
        Args:
            input_dim: dimensione input (13 features ECG)
            hidden_dims: lista di dimensioni per hidden layers (es. [20, 16])
                        Se None, usa [20] per compatibilità backward
            embedding_dim: dimensione embedding output (32)
            dropout: dropout rate per ogni layer
            normalize: se True, normalizza embeddings a L2-norm=1
        """
        super(ECGEncoder, self).__init__()
        self.normalize = normalize
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        # Default a singolo hidden layer se non specificato
        if hidden_dims is None:
            hidden_dims = [20]

        self.hidden_dims = hidden_dims

        # Costruisci layers dinamicamente
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Activation (ELU)
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (embedding)
        layers.append(nn.Linear(prev_dim, embedding_dim))
        layers.append(nn.BatchNorm1d(embedding_dim))
        # NO activation dopo ultimo layer (per embeddings)
        # NO dropout dopo embedding layer (non serve)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) input features
        Returns:
            embeddings: (batch_size, embedding_dim) L2-normalized embeddings
        """
        x = self.network(x)

        # L2 normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x


class ECGEncoderWithMomentum(nn.Module):
    """
    ECG Encoder con momentum updates.
    Utile per embedding memory bank o momentum contrast learning.
    """

    def __init__(self, input_dim: int = 13, hidden_dims: List[int] = None,
                 embedding_dim: int = 32, dropout: float = 0.2, normalize: bool = True):
        """
        Args:
            input_dim: dimensione input
            hidden_dims: lista di dimensioni per hidden layers
            embedding_dim: dimensione embedding output
            dropout: dropout rate
            normalize: L2 normalization
        """
        super(ECGEncoderWithMomentum, self).__init__()
        self.encoder = ECGEncoder(input_dim, hidden_dims, embedding_dim, dropout, normalize)
        self.momentum_encoder = ECGEncoder(input_dim, hidden_dims, embedding_dim, dropout, normalize)

        # Disabilita gradients per momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con encoder principale"""
        return self.encoder(x)

    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con momentum encoder"""
        with torch.no_grad():
            return self.momentum_encoder(x)

    def update_momentum(self, momentum: float = 0.999):
        """
        Aggiorna momentum encoder con media mobile.
        Args:
            momentum: coefficiente di aggiornamento (default 0.999)
        """
        with torch.no_grad():
            for param, momentum_param in zip(self.encoder.parameters(),
                                            self.momentum_encoder.parameters()):
                momentum_param.data = momentum_param.data * momentum + param.data * (1 - momentum)
