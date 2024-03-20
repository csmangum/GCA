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

    def predict(self, current_state: torch.Tensor) -> list:
        """
        Predict the output of the network.

        Parameters
        ----------
        current_state : torch.Tensor
            N length tensor of the current state.

        Returns
        -------
        list
            The predicted next state.
        """
        new_state = []

        for i in range(len(current_state)):
            left = current_state[i - 1] if i > 0 else 0
            center = current_state[i]
            right = current_state[i + 1] if i < len(current_state) - 1 else 0
            next_state = self.forward(
                torch.tensor([left, center, right], dtype=torch.float32).view(
                    1, -1
                )  # Reshape to add batch dimension
            )
            new_state.append(int(round(next_state.item())))
        return new_state
