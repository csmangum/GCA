import numpy as np
import torch
import torch.nn as nn


class ArtificialEvolution:
    """
    Class to perform artificial evolution on a population of neural networks.

    Parameters
    ----------
    model : nn.Module
        Neural network model to evolve.
    population : int
        Number of networks in the population.
    parents : int
        Number of parents to select from the population.
    mutation_rate : float, optional
        Probability of mutating a weight, by default 0.1.
    scale : float, optional
        Scale of the mutation, by default 0.05.

    Attributes
    ----------
    model : nn.Module
        Neural network model to evolve.
    criterion : nn.Module
        Loss function to use.
    population_size : int
        Number of networks in the population.
    population : list
        List of neural networks in the population.
    parents : int
        Number of parents to select from the population.
    mutation_rate : float
        Probability of mutating a weight.
    scale : float
        Scale of the mutation.
    population_history : list
        List of populations at each cycle.

    Methods
    -------
    initialize_population(size: int) -> list:
        Initialize a population of neural networks.
    evaluate_fitness(network: nn.Module, criterion: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        Evaluate the fitness of a neural network on a dataset.
    select_parents(population: list, fitnesses: list, num_parents: int) -> list:
        Select the best parents from the population based on fitness.
    crossover(parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        Create a child network by averaging the weights of two parent networks.
    mutate(network: nn.Module) -> None:
        Mutate the weights of a neural network in place.
    run(cycles: int, X: torch.Tensor, y: torch.Tensor, history: bool = True) -> tuple:
        Run the evolutionary training process.
    best(X: torch.Tensor, y: torch.Tensor) -> nn.Module:
        Return the best network from the population.
    """

    def __init__(
        self,
        model: nn.Module,
        population: int,
        parents: int,
        mutation_rate: float = 0.1,
        scale: float = 0.05,
    ) -> None:
        self.model = model
        self.criterion = nn.BCELoss()
        self.population_size = population
        self.population = self.initialize_population(population)
        self.parents = parents
        self.mutation_rate = mutation_rate
        self.scale = scale
        self.population_history = []

    def initialize_population(self, size: int) -> list:
        """
        Initialize a population of neural networks.

        Parameters
        ----------
        size : int
            Number of networks in the population.

        Returns
        -------
        list
            List of SimpleSequentialNetwork instances.
        """
        return [self.model() for _ in range(size)]

    def evaluate_fitness(
        self,
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

        with torch.no_grad():
            outputs = network(inputs)
            loss = criterion(outputs, targets)
        return -loss.item()  # Using negative loss as fitness, lower loss is better

    def select_parents(
        self, population: list, fitnesses: list, num_parents: int
    ) -> list:
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

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
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
        child = self.model()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            child_param.data.copy_((param1.data + param2.data) / 2.0)
        return child

    def mutate(self, network: nn.Module) -> None:
        """
        Mutate the weights of a neural network in place.

        Parameters
        ----------
        network : nn.Module
            Neural network to mutate.
        """
        for param in network.parameters():
            if torch.rand(1) < self.mutation_rate:
                param.data += self.scale * torch.randn_like(param)

    def run(
        self, cycles: int, X: torch.Tensor, y: torch.Tensor, history: bool = True
    ) -> tuple:
        """
        Run the evolutionary training process.

        First, evaluate the fitness of each network in the population.
        Then, select the best parents based on fitness.
        Create a new generation by crossing over and mutating the parents.
        Repeat for the specified number of cycles.

        Parameters
        ----------
        cycles : int
            Number of training cycles.
        X : torch.Tensor
            Input data of shape (batch_size, num_features).
        y : torch.Tensor
            Target labels of shape (batch_size, 1).
        history : bool, optional
            Whether to store the population history, by default True.

        Returns
        -------
        tuple
            Final population of neural networks and the best fit network.
        """

        for cycle in range(cycles):
            if history:
                self.population_history.append(self.population)
            fitnesses = [
                self.evaluate_fitness(net, self.criterion, X, y)
                for net in self.population
            ]
            parents = self.select_parents(self.population, fitnesses, self.parents)

            next_generation = []
            while len(next_generation) < self.population_size:
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        child = self.crossover(parents[i], parents[j])
                        self.mutate(child)
                        next_generation.append(child)
                        if len(next_generation) >= self.population_size:
                            break
            self.population = next_generation

            print(
                f"Cycle {cycle}: Average fitness: {np.mean(fitnesses):.2f} Best: {max(fitnesses):.2f} Worst: {min(fitnesses):.2f}"
            )
        if history:
            self.population_history.append(self.population)

        return self.population, self.best(X, y)

    def best(self, X: torch.Tensor, y: torch.Tensor) -> nn.Module:
        return max(
            self.population,
            key=lambda x: self.evaluate_fitness(x, self.criterion, X, y),
        )
