import torch
from torch import nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
    )


class GalaxyCNN(nn.Module):
    """Same architecture as project 1, with one input channel per SDSS band.

    4 conv blocks (width w, 2w, 4w, 8w), each followed by 2x2 max pooling,
    then global average pooling, dropout, and a linear layer.
    """

    def __init__(self, in_channels=5, width=32, dropout=0.3, n_classes=2):
        super().__init__()
        chans = [in_channels, width, 2 * width, 4 * width, 8 * width]
        layers = []
        for i in range(4):
            layers += [conv_block(chans[i], chans[i + 1]), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(chans[-1], n_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def split_at(self, k):
        """The network as two halves, so activations after layer k can be read and fed back in."""
        return self.features[:k], nn.Sequential(self.features[k:], self.head)


def build(hp, in_channels=5):
    return GalaxyCNN(in_channels=in_channels, width=hp["width"], dropout=hp["dropout"])


def save(model, hp, path, **extra):
    torch.save({"state_dict": model.state_dict(), "hparams": hp, **extra}, path)


def load(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build(ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
