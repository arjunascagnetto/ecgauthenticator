import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """Vanilla Autoencoder: 13D → 32D → 13D con hidden layer 20D"""

    def __init__(self, input_dim=13, latent_dim=32, hidden_dim=20,
                 encoder_dropout=0.2, decoder_dropout=0.1):
        """
        Args:
            input_dim: numero di features ECG (13)
            latent_dim: dimensione spazio latente (32)
            hidden_dim: dimensione hidden layer (20)
            encoder_dropout: probabilità dropout encoder
            decoder_dropout: probabilità dropout decoder
        """
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder: 13 → 20 → 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(encoder_dropout),

            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ELU()
        )

        # Decoder: 32 → 20 → 13
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(decoder_dropout),

            nn.Linear(hidden_dim, input_dim)
            # Output layer senza attivazione (ricostruzione lineare)
        )

        # Inizializzazione He per ELU
        self._init_weights()

    def _init_weights(self):
        """Inizializza pesi con He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        """Estrae l'embedding 32D"""
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
