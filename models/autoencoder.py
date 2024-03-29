import torch
import torch.nn as nn
import torch.optim as optim


class AutoEncoder(nn.Module):
    """
    Autoencoder class to compress the weights of a neural network into a
    3-dimensional tensor.

    The autoencoder consists of an encoder and a decoder. The encoder
    compresses the input data into a 3-dimensional tensor, and the decoder
    decompresses the 3-dimensional tensor back into the original input shape.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the autoencoder, first encoding the input and
        then decoding it
    encode_weights(weights: torch.Tensor, device: torch.device) -> torch.Tensor
        Encode the weights using the encoder layer into a 3-dimensional tensor
    train(data: torch.tensor, input_size: int, learning_rate: float = 0.001, save_as: str = None) -> nn.Module
        Training loop for the autoencoder
    """

    def __init__(self, input_size: int) -> None:
        """
        Initialize the autoencoder with the input size

        Parameters
        ----------
        input_size : int
            The size of the input
        """
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1064),
            nn.LeakyReLU(),
            nn.Linear(1064, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 1064),
            nn.LeakyReLU(),
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

    def encode_weights(
        self, weights: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Encode the weights using the encoder layer into a 3-dimensional tensor

        Parameters
        ----------
        weights : torch.Tensor
            The weights to encode
        device : torch.device
            The device to use, either "cuda" or "cpu"

        Returns
        -------
        torch.Tensor
            The encoded weights
        """
        with torch.no_grad():
            weights = weights.to(device)
            encoded = self.encoder(weights)
        return encoded

    @staticmethod
    def train(
        data: torch.tensor,
        input_size: int,
        learning_rate: float = 0.001,
        save_as: str = None,
    ) -> nn.Module:
        """
        Training loop for the autoencoder

        Parameters
        ----------
        data : torch.Tensor
            The input data to train the autoencoder on
        input_size : int
            The size of the input data
        save_as : str, optional
            The path to save the model to, by default None
        """
        model = AutoEncoder(input_size)
        criterion = nn.MSELoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        data = data.to(device)

        # Train the autoencoder
        epoch = 0
        while True:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

            epoch += 1

            # Break if the loss is below the threshold
            if loss.item() < 0.01:
                break

        if save_as:
            torch.save(model.state_dict(), f"{save_as}.pt")

        return model, device
