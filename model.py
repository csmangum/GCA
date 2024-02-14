import torch
import torch.nn as nn


class Rule30CNN(nn.Module):
    def __init__(self, num_cells: int = 101) -> None:
        """
        Initialize the Rule30CNN model.

        Parameters
        ----------
        num_cells : int
            The number of cells in each row of the cellular automaton.
        """
        super(Rule30CNN, self).__init__()
        self.num_cells = num_cells
        self.conv1 = nn.Conv1d(1, 10, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10 * num_cells, num_cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Rule30CNN model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the model.

        Returns
        -------
        torch.Tensor
            The output tensor from the model.
        """
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 10 * self.num_cells)
        x = torch.sigmoid(self.fc1(x))
        return x
