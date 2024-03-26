import torch
import torch.nn as nn
import torch.optim as optim


class WeightAutoencoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        """
        Initialize the autoencoder with the input size

        Parameters
        ----------
        input_size : int
            The size of the input
        """
        super(WeightAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1064),
            nn.ReLU(),
            nn.Linear(1064, 532),
            nn.ReLU(),
            nn.Linear(532, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 532),
            nn.ReLU(),
            nn.Linear(532, 1064),
            nn.ReLU(),
            nn.Linear(1064, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the autoencoder, first encoding the input and
        then decoding it

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def encode_weights(autoencoder, weights):
    """
    Encode the weights using the autoencoder

    Parameters
    ----------
    autoencoder : nn.Module
        The autoencoder to use
    weights : torch.Tensor
        The weights to encode

    Returns
    -------
    torch.Tensor
        The encoded weights
    """
    with torch.no_grad():
        return autoencoder.encoder(weights)


def train(data: torch.tensor, num_epochs: int, input_size: int) -> WeightAutoencoder:
    """
    Training loop for the autoencoder

    Parameters
    ----------
    data : torch.Tensor
        The input data
    num_epochs : int
        The number of epochs to train the model
    """
    autoencoder = WeightAutoencoder(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Train the autoencoder
    epoch = 0
    while True:
        optimizer.zero_grad()
        outputs = autoencoder(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        epoch += 1

        if loss.item() < 0.01:
            break

    torch.save(autoencoder.state_dict(), "autoencoder.pth")
