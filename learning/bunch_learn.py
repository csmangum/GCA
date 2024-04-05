import torch
import torch.nn as nn
import torch.optim as optim

from models.simple import SimpleSequentialNetwork
from util import execute_rule, extract_parameters


def train(
    epochs: int,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    rule_number: int,
    verbose: bool = True,
) -> tuple:
    """
    Train the model.

    Parameters
    ----------
    epochs : int
        The number of epochs to train the model.
    model : nn.Module
        The model to train.
    criterion : nn.Module
        The loss function.
    optimizer : optim.Optimizer
        The optimizer to use.
    X : torch.Tensor
        The input data.
    y : torch.Tensor
        The target data.
    rule_number : int
        The rule number to train on.
    verbose : bool
        Whether to print the training progress.

    Returns
    -------
    tuple
        A tuple containing the parameter snapshots, loss records, and gradient norms.
    """

    # Populating the parameters with initial values
    parameter_snapshots = []
    loss_records = []
    gradient_norms = []

    # Initial parameters and loss
    parameter_snapshots.append(extract_parameters(model))
    loss_records.append(criterion(model(X), y).item())

    for epoch in range(epochs):

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()

        # Log gradient norms for this batch
        layer_grad_norms = [
            torch.norm(param.grad).item()
            for param in model.parameters()
            if param.grad is not None
        ]
        gradient_norms.append(layer_grad_norms)

        optimizer.step()

        parameter_snapshots.append(extract_parameters(model))
        loss_records.append(loss.item())

        if epoch % 10 == 0 and verbose:
            print(f"Epoch {epoch} - Loss: {loss.item()}")

        if loss.item() < 0.01:
            break

    return parameter_snapshots, loss_records, gradient_norms


def bunch_learn(
    model_count: int,
    rule_number: int,
    learning_epochs: int,
    seed: int = None,
    verbose: bool = True,
) -> tuple:
    """
    Train a bunch of models on the same rule number.

    Parameters
    ----------
    model_count : int
        The number of models to train.
    rule_number : int
        The rule number to train on.
    learning_epochs : int
        The number of epochs to train each model.
    seed : int
        The seed to use for reproducibility.
    verbose : bool
        Whether to print the training progress.

    Returns
    -------
    tuple
        A tuple containing the parameter snapshots, loss records, and gradient norms.
    """

    total_snapshots = []
    total_loss_records = []
    total_gradient_norms = []

    for i in range(model_count):

        cell_states = execute_rule(rule_number)

        X = torch.tensor(
            [[i >> 2, (i >> 1) & 1, i & 1] for i in range(8)], dtype=torch.float32
        )

        y = torch.tensor(cell_states, dtype=torch.float32).view(-1, 1)

        if seed:
            torch.manual_seed(seed)

        learning = SimpleSequentialNetwork()

        criterion = nn.BCELoss()

        optimizer = optim.Adam(learning.parameters(), lr=0.01)

        parameter_snapshots, loss_records, gradient_norms = train(
            learning_epochs, learning, criterion, optimizer, X, y, i, verbose
        )

        total_snapshots.append(parameter_snapshots)
        total_loss_records.append(loss_records)
        total_gradient_norms.append(gradient_norms)

    return learning, total_snapshots, total_loss_records, total_gradient_norms
