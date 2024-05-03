import torch
from torch import nn


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """
        Simple shallow transformer model that takes fixed index of 3 features and predicts the
        next state which is a boolean value

        Parameters
        ----------
        input_dim : int
            The number of input features
        hidden_dim : int
            The number of hidden units
        output_dim : int
            The number of output features
        """
        super(SimpleTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        x = self.transformer(x)
        x = self.fc(x)
        return x
