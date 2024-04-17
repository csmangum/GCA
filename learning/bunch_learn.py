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
    early_stopping: bool = True,
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
    early_stopping : bool
        Whether to stop training when the loss is below 0.01.

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

        if loss.item() < 0.01 and early_stopping:
            break

    return parameter_snapshots, loss_records, gradient_norms


def bunch_learn(
    model_count: int,
    rule_number: int,
    max_epochs: int,
    verbose: bool = True,
    seed: int = None,
    learning_rate: float = 0.01,
    early_stopping: bool = True,
) -> dict:
    """
    Train a bunch of models on the same rule number.

    Parameters
    ----------
    model_count : int
        The number of models to train.
    rule_number : int
        The rule number to train on.
    max_epochs : int
        The max number of epochs to train each model, otherwise, the training
        stops when the loss is below 0.01.
    verbose : bool
        Whether to print the training progress.
    seed : int
        The random seed to use.
    learning_rate : float
        The learning rate to use.
    early_stopping : bool
        Whether to stop training when the loss is below 0.01.

    Returns
    -------
    dict
        A dictionary containing the training results. Including the parameter
        snapshots, loss records, and gradient norms. Indexed by the seed.
    """

    training_results = {}

    for i in range(model_count):

        # Random seed
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = torch.randint(0, 1000000, (1,)).item()
            torch.manual_seed(seed)

        cell_states = execute_rule(rule_number)

        X = torch.tensor(
            [[i >> 2, (i >> 1) & 1, i & 1] for i in range(8)], dtype=torch.float32
        )

        y = torch.tensor(cell_states, dtype=torch.float32).view(-1, 1)

        learning = SimpleSequentialNetwork()

        criterion = nn.BCELoss()  # Binary Cross Entropy Loss

        optimizer = optim.Adam(learning.parameters(), lr=learning_rate)

        parameter_snapshots, loss_records, gradient_norms = train(
            max_epochs, learning, criterion, optimizer, X, y, i, verbose, early_stopping
        )

        results = {
            "snapshots": parameter_snapshots,
            "losses": loss_records,
            "gradients": gradient_norms,
            "learning_rate": learning_rate,
            "rule_number": rule_number,
            "model_number": i,
        }
        training_results[seed] = results

    return training_results
