"""
Simple example of using evolutionary algorithms to train a neural network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = (iris.target == 0).astype(np.float32)  # 1 if Setosa, 0 otherwise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define the Neural Network for binary classification
class BinaryClassifier(nn.Module):
    """
    Simple feedforward neural network with 1 hidden layer.

    Input: 4 features
    Hidden layer: 10 units
    Output: 1 unit with sigmoid activation
    """

    def __init__(self) -> None:
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1), with values between 0 and 1
            indicating the probability of class 1.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Initialize Population
def initialize_population(size: int) -> list:
    """
    Initialize a population of neural networks.

    Parameters
    ----------
    size : int
        Number of networks in the population.

    Returns
    -------
    list
        List of BinaryClassifier instances.
    """
    return [BinaryClassifier() for _ in range(size)]


# Evaluate Fitness
def evaluate_fitness(
    network: nn.Module,
    criterion: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Evaluate the fitness of a neural network on a dataset.

    Parameters
    ----------
    network : nn.Module
        Neural network to evaluate.
    criterion : nn.Module
        Loss function to use.
    inputs : torch.Tensor
        Input data of shape (batch_size, num_features).
    targets : torch.Tensor
        Target labels of shape (batch_size, 1).

    Returns
    -------
    float
        Negative loss value as fitness. Lower loss is better.
    """
    network.eval()
    with torch.no_grad():
        outputs = network(inputs)
        loss = criterion(outputs, targets)
    return -loss.item()  # Using negative loss as fitness, lower loss is better


# Select Parents
def select_parents(population: list, fitnesses: list, num_parents: int) -> list:
    """
    Select the best parents from the population based on fitness.

    Parameters
    ----------
    population : list
        List of neural networks.
    fitnesses : list
        List of fitness values corresponding to each network in the population.
    num_parents : int
        Number of parents to select.

    Returns
    -------
    list
        List of selected parents (neural networks).
    """
    parents = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [parent for parent, _ in parents[:num_parents]]


# Crossover
def crossover(parent1: nn.Module, parent2: nn.Module) -> nn.Module:
    """
    Create a child network by averaging the weights of two parent networks.

    Parameters
    ----------
    parent1 : nn.Module
        First parent network.
    parent2 : nn.Module
        Second parent network.

    Returns
    -------
    nn.Module
        Child network with averaged weights.
    """
    child = BinaryClassifier()
    for child_param, param1, param2 in zip(
        child.parameters(), parent1.parameters(), parent2.parameters()
    ):
        child_param.data.copy_((param1.data + param2.data) / 2.0)
    return child


# Mutation
def mutate(network: nn.Module, mutation_rate: float = 0.1, scale: float = 0.05) -> None:
    """
    Mutate the weights of a neural network in place.

    Parameters
    ----------
    network : nn.Module
        Neural network to mutate.
    mutation_rate : float, optional
        Probability of mutating each parameter, by default 0.1.
    scale : float, optional
        Scale of the mutation, by default 0.05.
    """
    for param in network.parameters():
        if torch.rand(1) < mutation_rate:
            param.data += scale * torch.randn_like(param)


# Run Evolution
def run_evolution(
    cycles: int, population_size: int, num_parents: int, mutation_rate: float = 0.1
) -> tuple:
    """
    Run the evolutionary training process.

    Parameters
    ----------
    cycles : int
        Number of cycles to run.
    population_size : int
        Number of networks in the population.
    num_parents : int
        Number of parents to select for each generation.
    mutation_rate : float, optional
        Probability of mutating each parameter, by default 0.1.

    Returns
    -------
    tuple
        Tuple of the final population and the loss function used.
    """
    criterion = nn.MSELoss()
    population = initialize_population(population_size)

    for cycle in range(cycles):
        fitnesses = [
            evaluate_fitness(net, criterion, X_train, y_train) for net in population
        ]
        parents = select_parents(population, fitnesses, num_parents)

        next_generation = []
        while len(next_generation) < population_size:
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    child = crossover(parents[i], parents[j])
                    mutate(child, mutation_rate)
                    next_generation.append(child)
                    if len(next_generation) >= population_size:
                        break
        population = next_generation
        print(f"Cycle {cycle}: Best fitness: {max(fitnesses)}")
        # Optionally: Print average fitness of the population
        print(
            "Cycle {}: Average fitness: {:.2f}".format(
                cycle, sum(fitnesses) / len(fitnesses)
            )
        )

    return population, criterion


# Parameters
cycles = 400
population_size = 10
num_parents = 5

# Start the evolutionary training
population, criterion = run_evolution(cycles, population_size, num_parents)


def evaluate_model(
    population: list, X_test: torch.Tensor, y_test: torch.Tensor
) -> None:
    """
    Evaluate the best network in the population on the test set.

    Parameters
    ----------
    population : list
        List of neural networks.
    X_test : torch.Tensor
        Test input data of shape (batch_size, num_features).
    y_test : torch.Tensor
        Test target labels of shape (batch_size, 1).
    """
    # Final evaluation on test set
    best_network = max(
        population, key=lambda net: evaluate_fitness(net, criterion, X_train, y_train)
    )
    best_network.eval()
    with torch.no_grad():
        outputs = best_network(X_test)
        loss = criterion(outputs, y_test)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_test).float().mean()

    print("Test set loss: {:.4f}".format(loss.item()))
    print("Test set accuracy: {:.2f}".format(accuracy.item()))


evaluate_model(population, X_test, y_test)
