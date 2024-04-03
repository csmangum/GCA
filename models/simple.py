from typing import Tuple

import torch
import torch.nn as nn


class SimpleSequentialNetwork(nn.Module):
    """
    A simple sequential network with 3 input nodes, 10 hidden nodes, and 1 output node.

    Methods
    -------
    forward(x)
        Forward pass of the network.
    predict(current_state)
        Predict the output of the network by passing the current state of a row
        of CA cells through the network.
    """

    def __init__(self) -> None:
        super(SimpleSequentialNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
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
        Predict the output of the network by passing the current state of a
        row of CA cells through the network.

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
            state_tensor = torch.tensor(
                [left, center, right], dtype=torch.float32
            ).view(1, -1)
            next_state = self.forward(state_tensor)
            new_state.append(int(round(next_state.item())))  # Round to 0 or 1
        return new_state

    @staticmethod
    def train(
        X: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float = 0.001,
        save_as: str = None,
    ) -> Tuple[nn.Module, list]:
        """
        Train the network on the given data.

        Parameters
        ----------
        X : torch.Tensor
            The input data.
        y : torch.Tensor
            The target data.
        input_size : int
            The size of the input.
        learning_rate : float
            The learning rate to use for training.
        save_as : str
            The file to save the trained model to.

        Returns
        -------
        nn.Module, list
            The trained model and the loss history.

        """
        model = SimpleSequentialNetwork()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        epoch = 0
        loss_history = []

        while True:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            print(f"Epoch {epoch}, Loss: {loss.item()}")

            epoch += 1

            if loss.item() < 0.01:
                break

        if save_as:
            torch.save(model.state_dict(), f"{save_as}.pt")

        return model, loss_history


class SimpleAllRules(SimpleSequentialNetwork):

    def __init__(self) -> None:
        super(SimpleAllRules, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )
