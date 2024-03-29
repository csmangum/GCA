from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class VariationalAutoEncoder(nn.Module):
    """
    Variational autoencoder class to compress the weights of a neural network
    into a 3-dimensional tensor.

    The variational autoencoder consists of an encoder and a decoder. The encoder
    compresses the input data into a 3-dimensional tensor, and the decoder
    decompresses the 3-dimensional tensor back into the original input shape.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the variational autoencoder, first encoding the input and
        then decoding it
    encode_weights(weights: torch.Tensor) -> torch.Tensor
        Encode the weights using the encoder layer into a 3-dimensional tensor
    loss_function(x: torch.Tensor, decoded: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor
        Calculate the loss function for the variational autoencoder by summing
        the mean squared error and the Kullback-Leibler divergence
    train(data: torch.tensor, input_size: int, learning_rate: float = 0.001, save_as: str = None) -> nn.Module
        Training loop for the variational autoencoder
    """

    def __init__(self, input_size: int) -> None:
        """
        Initialize the variational autoencoder with the input size

        Parameters
        ----------
        input_size : int
            The size of the input
        """
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1064),
            nn.ReLU(),
            nn.Linear(1064, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 1064),
            nn.ReLU(),
            nn.Linear(1064, input_size),
        )
        self.mu = nn.Linear(3, 3)
        self.logvar = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the variational autoencoder, first encoding the input and
        then decoding it. Also returns the mean and log variance of the encoded
        tensor.

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
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        decoded = self.decoder(z)
        return decoded, mu, logvar

    def encode_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Encode the weights using the encoder layer into a 3-dimensional tensor

        Parameters
        ----------
        weights : torch.Tensor
            The weights to encode

        Returns
        -------
        torch.Tensor
            The encoded weights
        """
        encoded = self.encoder(weights)
        return encoded

    def loss_function(
        self,
        x: torch.Tensor,
        decoded: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the loss function for the variational autoencoder by summing
        the mean squared error and the Kullback-Leibler divergence.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor
        decoded : torch.Tensor
            The decoded tensor
        mu : torch.Tensor
            The mean tensor
        logvar : torch.Tensor
            The log variance tensor

        Returns
        -------
        torch.Tensor
            The loss value
        """
        BCE = nn.functional.mse_loss(decoded, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    @staticmethod
    def train(
        data: torch.tensor,
        input_size: int,
        learning_rate: float = 0.001,
        save_as: str = None,
    ) -> Tuple[nn.Module, list]:
        """
        Training loop for the variational autoencoder

        Parameters
        ----------
        data : torch.tensor
            The data to train on
        input_size : int
            The size of the input
        learning_rate : float, optional
            The learning rate for the optimizer, by default 0.001
        save_as : str, optional
            The name to save the model as, by default None

        Returns
        -------
        nn.Module, list
            The trained model and the loss history
        """

        model = VariationalAutoEncoder(input_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        data = data.to(device)

        epoch = 0
        loss_history = []

        while True:
            optimizer.zero_grad()
            decoded, mu, logvar = model(data)
            loss = model.loss_function(data, decoded, mu, logvar)
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            if loss.item() < 0.01:
                break

            epoch += 1

        if save_as:
            torch.save(model.state_dict(), f"{save_as}.pt")

        return model, loss_history
