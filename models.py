import torch
import torch.nn as nn


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
    A 3-layer convolutional autoencoder
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

    def loss_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((x - y) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        z = self.encoder(x)
        # Decode
        y = self.decoder(z)

        return y


if __name__ == '__main__':
    # Test autoencoder
    net = Autoencoder(latent_dim=128, num_layers=3, width=32)
    print(net)
    x = torch.randn(1, 1, 64, 64)
    y = net(x)
    print(y.shape)
