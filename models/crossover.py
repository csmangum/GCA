from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        raise NotImplementedError(
            "Crossover strategy must implement the crossover method."
        )


class AverageCrossover(CrossoverStrategy):
    """
    Crossover strategy that averages the weights of two parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            child_param.data.copy_((param1.data + param2.data) / 2.0)
        return child


class RandomCrossover(CrossoverStrategy):
    """
    Crossover strategy that randomly selects weights from two parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            mask = torch.rand(param1.size()) > 0.5
            child_param.data.copy_(torch.where(mask, param1.data, param2.data))
        return child


class RandomPointCrossover(CrossoverStrategy):
    """
    Crossover strategy that randomly selects a point and swaps weights from two
    parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            mask = torch.zeros(param1.size())
            mask[:, : param1.size(1) // 2] = 1
            child_param.data.copy_(torch.where(mask, param1.data, param2.data))
        return child


class RandomRangeCrossover(CrossoverStrategy):
    """
    Crossover strategy that randomly selects a range and swaps weights from two
    parent networks.
    """

    def crossover(self, parent1: nn.Module, parent2: nn.Module) -> nn.Module:
        child = type(parent1)()
        for child_param, param1, param2 in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            mask = torch.zeros(param1.size())
            start = torch.randint(0, param1.size(1), (1,))
            end = torch.randint(start, param1.size(1), (1,))
            mask[:, start:end] = 1
            child_param.data.copy_(torch.where(mask, param1.data, param2.data))
        return child
