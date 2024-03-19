import torch
import torch.nn as nn


class SimpleSequentialNetwork(nn.Module):
    """
    A simple sequential network with 3 input nodes, 10 hidden nodes, and 1 output node.

    Methods
    -------
    forward(x)
        Forward pass of the network.
    """

    def __init__(self) -> None:
        super(SimpleSequentialNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the network.

        Returns
        -------
        torch.Tensor
            The output of the network.
        """
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input to the network.

        Returns
        -------
        torch.Tensor
            The predicted output of the network.
        """
        return self.forward(x)
