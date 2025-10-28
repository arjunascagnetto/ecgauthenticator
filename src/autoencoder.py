import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """Vanilla Autoencoder: 13D → 16D → 13D"""

    def __init__(self, input_dim=13, latent_dim=16, dropout_rate=0.4):
        """
        Args:
            input_dim: numero di features ECG (13)
            latent_dim: dimensione spazio latente (16)
            dropout_rate: probabilità dropout
        """
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: 13 → 16 (diretto)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Decoder: 16 → 13 (diretto)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
            # Output layer senza attivazione (ricostruzione lineare)
        )

    def encode(self, x):
        """Estrae l'embedding 16D"""
        return self.encoder(x)

    def decode(self, z):
        """Ricostruisce dai latent embedding"""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass: x → encoder → decoder → x_reconstructed"""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class AutoencoderLoss(nn.Module):
    """Loss: MSE + L2 regularization sui pesi"""

    def __init__(self, model, lambda_l2=1e-4):
        """
        Args:
            model: istanza Autoencoder
            lambda_l2: peso regolarizzazione L2
        """
        super(AutoencoderLoss, self).__init__()
        self.model = model
        self.lambda_l2 = lambda_l2
        self.mse = nn.MSELoss()

    def forward(self, x, x_reconstructed):
        """
        Calcola loss totale: MSE + L2 regularization

        Args:
            x: input originale
            x_reconstructed: output del modello

        Returns:
            loss_total, loss_mse, loss_l2
        """
        # Reconstruction loss
        loss_mse = self.mse(x, x_reconstructed)

        # L2 regularization sui pesi
        loss_l2 = torch.tensor(0.0, device=x.device)
        for param in self.model.parameters():
            loss_l2 += torch.sum(param ** 2)

        loss_l2 = self.lambda_l2 * loss_l2

        # Loss totale
        loss_total = loss_mse + loss_l2

        return loss_total, loss_mse, loss_l2
