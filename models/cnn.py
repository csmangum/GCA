import torch
import torch.nn as nn


class AutomataCNN(nn.Module):
    """
    A convolutional neural network for learning 1D cellular automata rules.

    The model takes as input a tensor of shape (batch_size, 1, num_cells) and
    returns a tensor of shape (batch_size, num_cells) containing the next state
    of the cellular automaton.

    The model consists of a single convolutional layer followed by a fully
    connected layer.

    Attributes
    ----------
    num_cells : int
        The number of cells in each row of the cellular automaton.
    conv1 : nn.Conv1d
        The first convolutional layer.
    fc1 : nn.Linear
        The first fully connected layer.

    Methods
    -------
    forward(x: torch.Tensor) -> torch.Tensor
        Forward pass of the Rule30CNN model.
    save(path: str) -> None
        Save the model to file.
    load(path: str) -> None
        Load the model from file.
    """

    def __init__(self, num_cells: int = 101) -> None:
        """
        Initialize the Rule30CNN model.

        Parameters
        ----------
        num_cells : int
            The number of cells in each row of the cellular automaton.
        """
        super(AutomataCNN, self).__init__()
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

    def save(self, path: str) -> None:
        """
        Save the model to file.

        Parameters
        ----------
        path : str
            The path to save the model to.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Load the model from file.
        """
        self.load_state_dict(torch.load(path))
