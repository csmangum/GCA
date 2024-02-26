"""
Kernel size:  
    Size of the sliding window that moves across the input signal. 
    For a 1D convolutional neural network (CNN), the kernel size is a single 
    integer that defines the length of this window. For a 1D CA CNN, the kernel
    size is typically 3, which means the window is 3 cells wide, for the 3 cell 
    states.
    
Padding:
    The number of cells to add to the beginning and end of the input signal. 
    This is useful for maintaining the same input and output dimensions. 
"""

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
        # Apply RELU activation function to the output of the first conv layer
        x = torch.relu(self.conv1(x))

        # Reshape the tensor to have the same number of columns as the number of cells
        x = x.view(-1, 10 * self.num_cells)

        # Apply the sigmoid activation function to the output of the first
        # fully connected layer
        x = torch.sigmoid(self.fc1(x))

        return x


class GeneralAutomataCNN(nn.Module):
    """
    A convolutional neural network for learning all 256 rules of 1D cellular automata.

    This model takes as input a tensor of shape (batch_size, 2, num_cells) where the first channel
    is the current state of the cellular automaton, and the second channel is the binary encoding of the rule.
    It returns a tensor of shape (batch_size, num_cells) containing the next state of the cellular automaton.

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
        Forward pass of the GeneralAutomataCNN model.
    """

    def __init__(self, num_cells: int = 101) -> None:
        """
        Parameters
        ----------
        num_cells : int
            The number of cells in each row of the cellular automaton.
        """
        super(GeneralAutomataCNN, self).__init__()
        self.num_cells = num_cells
        # Adjusted for 2 input channels: the current state and the rule encoding
        self.conv1 = nn.Conv1d(2, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * num_cells, num_cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GeneralAutomataCNN model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the model.

        Returns
        -------
        torch.Tensor
            The output tensor from the model.
        """
        # Apply RELU activation function to the output of the first conv layer
        x = torch.relu(self.conv1(x))

        # Reshape the tensor to have the same number of columns as the number of cells
        x = x.view(-1, 256 * self.num_cells)

        # Apply the sigmoid activation function to the output of the first
        # fully connected layer
        x = torch.sigmoid(self.fc1(x))

        return x
