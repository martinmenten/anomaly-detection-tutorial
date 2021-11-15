import torch
import torch.nn as nn
from torch import Tensor


class Reshape(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.size)


def build_encoder(in_channels: int, num_layers: int, width: int) -> nn.Module:
    # Build encoder
    hidden_dims = [width * 2 ** i for i in range(num_layers)]
    encoder = []
    for h_dim in hidden_dims:
        encoder.append(
            nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
        )
        in_channels = h_dim
    return nn.Sequential(*encoder)


def build_decoder(out_channels: int, num_layers: int, width: int) -> nn.Module:
    # Build decoder
    hidden_dims = [width * 2 ** i for i in range(num_layers)]
    decoder = []
    for i in range(len(hidden_dims) - 1, 0, -1):
        decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i - 1],
                                   kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                nn.BatchNorm2d(hidden_dims[i - 1]),
                nn.LeakyReLU(),
            )
        )
    decoder.append(
        nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0],
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.LeakyReLU(),
        )
    )
    # Final layer
    decoder.append(
        nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[0], 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    )
    return nn.Sequential(*decoder)


class Autoencoder(nn.Module):
    """
    A n-layer convolutional autoencoder
    """
    def __init__(self,
                 latent_dim: int,
                 img_size: int = 64,
                 num_layers: int = 3,
                 width: int = 32):
        super().__init__()

        hidden_dims = [width * 2 ** i for i in range(num_layers)]
        intermediate_res = img_size // 2 ** num_layers
        intermediate_feats = intermediate_res * intermediate_res * hidden_dims[-1]

        # Build encoder
        self.encoder = build_encoder(1, num_layers, width)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intermediate_feats, latent_dim),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, intermediate_feats),
            Reshape((-1, hidden_dims[-1], intermediate_res, intermediate_res)),
        )

        # Build decoder
        self.decoder = build_decoder(1, num_layers, width)

    def loss_function(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.mean((x - y) ** 2)

    def forward(self, x: Tensor) -> Tensor:
        # Encode
        res = self.encoder(x)
        # Bottleneck
        z = self.bottleneck(res)
        decoder_inp = self.decoder_input(z)
        # Decode
        y = self.decoder(decoder_inp)
        return y


class VAE(nn.Module):
    """
    A n-layer variational autoencoder
    adapted from: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    """
    def __init__(self,
                 latent_dim: int,
                 img_size: int = 64,
                 num_layers: int = 3,
                 width: int = 32):
        super().__init__()

        hidden_dims = [width * 2 ** i for i in range(num_layers)]
        intermediate_res = img_size // 2 ** num_layers
        intermediate_feats = intermediate_res * intermediate_res * hidden_dims[-1]

        # Build encoder
        self.encoder = build_encoder(1, num_layers, width)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(intermediate_feats, latent_dim * 2),
        )
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, intermediate_feats),
            Reshape((-1, hidden_dims[-1], intermediate_res, intermediate_res)),
        )

        # Build decoder
        self.decoder = build_decoder(1, num_layers, width)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: Mean of the estimated latent Gaussian
        :param logvar: Standard deviation of the estimated latent Gaussian
        """
        unit_gaussian = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return unit_gaussian * std + mu

    def loss_function(self, x: Tensor, y: Tensor, mu: Tensor, logvar: Tensor,
                      kl_weight: float = 1.0) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param x: Input image
        :param y: Reconstructed image
        :param mu: Mean of the estimated latent Gaussian
        :param logvar: Standard deviation of the estimated latent Gaussian
        :param kl_weight: Account for the minibatch size from the dataset
        """
        recon_loss = torch.mean((x - y) ** 2)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        loss = recon_loss + kl_weight * kl_loss
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

    def anomaly_score(self, x: Tensor) -> Tensor:
        """
        Computes the anomaly score (kl-divergence)
        """
        res = self.encoder(x)
        mu, logvar = torch.chunk(self.bottleneck(res), 2, dim=1)
        return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # Encode
        res = self.encoder(x)
        # Bottleneck
        mu, logvar = torch.chunk(self.bottleneck(res), 2, dim=1)
        z = self.reparameterize(mu, logvar)
        decoder_inp = self.decoder_input(z)
        # Decode
        y = self.decoder(decoder_inp)
        return y, mu, logvar


if __name__ == '__main__':
    # Test autoencoder
    net = Autoencoder(latent_dim=128, num_layers=3, width=32)
    print(net)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = net(x)
    print(y.shape)

    # Test VAE
    net = VAE(latent_dim=128, num_layers=3, width=32)
    print(net)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y, mu, logvar = net(x)
    print(y.shape, mu.shape, logvar.shape)
