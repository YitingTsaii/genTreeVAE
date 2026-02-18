# ==========================================
# VARIATIONAL AUTOENCODER
# ==========================================

import torch
from torch import nn
from .fmatrix import FMatrixLayer


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Phylogenetic F-Matrices.

    Architecture:
    1. Encoder: Compresses the input F-matrix (flat vector) into a latent distribution
       parameterized by mean (mu) and log-variance (logvar).
    2. Sampling: Uses the reparameterization trick to sample a latent vector 'z'.
    3. Decoder: Reconstructs the flat vector from 'z'.
    4. Projection: Passes the raw reconstruction through 'FMatrixLayer' to enforce
       perfect phylogeny constraints (the 4-point condition).

    Attributes:
        input_dim (int): Size of the flattened input vector (Deep Lower Triangle).
        hidden_dims (list): List of integers defining the size of hidden layers.
        latent_dim (int): Size of the bottleneck latent space (z).
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, device):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.device = device

        # Encoder
        encoder_blocks = []
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            encoder_blocks.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_blocks)

        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Decoder
        decoder_blocks = []
        reversed_hidden = self.hidden_dims[::-1]
        prev_dim = self.latent_dim
        for h_dim in reversed_hidden:
            decoder_blocks.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_blocks)

        self.fc_final = nn.Linear(self.hidden_dims[0], self.input_dim)

        # Projection Layer
        self.f_matrix_layer = FMatrixLayer(input_dim, device)

    def encode(self, x):
        """
        Passes input through encoder to get latent distribution parameters.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        The 'Reparameterization Trick':
        Instead of sampling z ~ N(mu, std) directly (which breaks backpropagation),
        we sample noise eps ~ N(0, 1) and transform it: z = mu + eps * std.
        This makes the sampling process differentiable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Reconstructs the input from the latent vector z.
        """
        h = self.decoder(z)
        out = self.fc_final(h)
        out = torch.sigmoid(out)  # Ensure 0-1 range initially

        # Apply projection
        out_projected, struct_loss = self.f_matrix_layer(out)
        return out_projected, struct_loss

    def forward(self, x):
        """
        Full forward pass: Encode -> Sample -> Decode.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, struct_loss = self.decode(z)
        return recon_x, mu, logvar, struct_loss
