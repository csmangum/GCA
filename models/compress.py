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
            nn.Linear(1064, 3),  # Encoding the input down to 2 dimensions
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 1064), nn.ReLU(), nn.Linear(1064, input_size)
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


def train(model: nn.Module, data: torch.Tensor, num_epochs: int = 1000) -> None:
    """
    Training loop for the autoencoder

    Parameters
    ----------
    model : nn.Module
        The autoencoder model
    data : torch.Tensor
        The input data
    num_epochs : int
        The number of epochs to train the model
    """
    input_size = len(data[0])
    autoencoder = WeightAutoencoder(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

    # Train the autoencoder
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = autoencoder(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
