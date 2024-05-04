from typing import TYPE_CHECKING, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.crossover import AverageCrossover, CrossoverStrategy
from models.mutation import GaussianMutation, MutationStrategy
from models import ModelFactory


class ArtificialEvolution:
    """
    Class to perform artificial evolution on a population of neural networks.

    Parameters
    ----------
    model : nn.Module
        Neural network model to evolve.
    factory : ModelFactory
        Factory class for creating models on the fly.
    settings : dict
        Settings to pass to the model factory.
    population : int
        Number of networks in the population.
    parents : int
        Number of parents to select from the population.
    crossover_strategy : CrossoverStrategy, optional
        Crossover strategy to use, by default AverageCrossover.
    mutation_strategy : MutationStrategy, optional
        Mutation strategy to use, by default GaussianMutation.

    Attributes
    ----------
    model_factory : ModelFactory
        Factory class for creating models on the fly.
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
    population_history : list
        List of populations at each cycle.
    fitness_history : list
        List of fitness values at each cycle. Tuple of (min, max, avg).
    crossover_strategy : CrossoverStrategy
        Crossover strategy to use.
    mutation_strategy : MutationStrategy
        Mutation strategy to use.

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
    log_fitness(fitness: list) -> Tuple[float, float, float]:
        Calculates the Minimum, Maximum and Average fitness values for each generation.
    """

    def __init__(
        self,
        model: nn.Module,
        settings: dict,
        population: int,
        parents: int,
        crossover_strategy: CrossoverStrategy = AverageCrossover(),
        mutation_strategy: MutationStrategy = GaussianMutation(),
    ) -> None:
        self.model_factory = ModelFactory(model, settings)
        self.criterion = nn.BCELoss()
        self.population_size = population
        self.population = self.initialize_population(population)
        self.parents = parents
        self.population_history = []
        self.fitness_history = []
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy

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
        return [self.model_factory() for _ in range(size)]

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
        Create a child network by the crossover of two parent networks.

        Parameters
        ----------
        parent1 : nn.Module
            First parent network.
        parent2 : nn.Module
            Second parent network.

        Returns
        -------
        nn.Module
            Child network created by crossover of the parents.
        """
        return self.crossover_strategy.crossover(parent1, parent2)

    def mutate(self, network: nn.Module) -> None:
        """
        Mutate the weights of a neural network in place.

        Parameters
        ----------
        network : nn.Module
            Neural network to mutate.
        """
        self.mutation_strategy.mutate(network)

    def run(
        self,
        max_cycles: int,
        X: torch.Tensor,
        y: torch.Tensor,
        history: bool = True,
        early_stopping: bool = False,
    ) -> tuple:
        """
        Run the evolutionary training process.

        First, evaluate the fitness of each network in the population.
        Then, select the best parents based on fitness.
        Create a new generation by crossing over and mutating the parents.
        Repeat for the specified number of cycles.

        Parameters
        ----------
        max_cycles : int
            Max number of training cycles.
        X : torch.Tensor
            Input data of shape (batch_size, num_features).
        y : torch.Tensor
            Target labels of shape (batch_size, 1).
        history : bool, optional
            Whether to store the population history, by default True.
        early_stopping : bool, optional
            Whether to stop early if the best fitness is above -0.02, by default False.

        Returns
        -------
        tuple
            Final population of neural networks and the best fit network.
        """

        for cycle in range(max_cycles):
            if history:
                self.population_history.append(self.population.copy())
            fitnesses = [
                self.evaluate_fitness(net, self.criterion, X, y)
                for net in self.population
            ]
            self.fitness_history.append(self.log_fitness(fitnesses))
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

            if max(fitnesses) > -0.02 and early_stopping:
                print(f"Early stopping at cycle {cycle}")
                break

        if history:
            self.population_history.append(self.population.copy())

        return self.population, self.best(X, y)

    def best(self, X: torch.Tensor, y: torch.Tensor) -> nn.Module:
        """
        Return the best network from the population.

        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (batch_size, num_features).
        y : torch.Tensor
            Target labels of shape (batch_size, 1).

        Returns
        -------
        nn.Module
            Best network from the population based on fitness
        """
        return max(
            self.population,
            key=lambda x: self.evaluate_fitness(x, self.criterion, X, y),
        )

    def log_fitness(self, fitness: list) -> Tuple[float, float, float]:
        """
        Stores the Minimum, Maximum and Average fitness values for each generation.

        Parameters
        ----------
        fitness : list
            List of fitness values for each network in the population.

        Returns
        -------
        Tuple[float, float, float]
            Minimum, Maximum and Average fitness values.
        """
        return min(fitness), max(fitness), np.mean(fitness)
