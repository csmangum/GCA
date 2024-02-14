from typing import Tuple

import numpy as np
import torch
from torch import nn

from automata import Automata
from model import Rule30CNN


class Learn:
    """
    A class to represent a machine learning model for a 1D cellular automaton.

    Attributes
    ----------
    rule_number : int
        The rule number to use for the cellular automaton.
    num_cells : int
        The number of cells in each row of the cellular automaton.
    num_generations : int
        The number of generations to generate.
    automata : Automata
        The cellular automaton to learn.
    model : Rule30CNN
        The machine learning model to train.
    criterion : nn.BCELoss
        The loss function to use for training.
    optimizer : torch.optim.Adam
        The optimizer to use for training.
    epoch : int
        The current epoch of training.
    loss : float
        The current loss value of the model.

    Methods
    -------
    generate_data(num_records)
        Generate training data for a 1D cellular automaton.
    prepare_data()
        Prepare training and test data for a 1D cellular automaton.
    early_stopping(actual_automata)
        Perform early stopping based on the accuracy of the model to the actual
        cellular automaton.
    train(epochs)
        Train a model to predict the next state of a 1D cellular automaton.
    """

    def __init__(
        self, rule_number: int = 30, num_cells: int = 101, num_generations: int = 100
    ):
        self.rule_number = rule_number
        self.num_cells = num_cells
        self.num_generations = num_generations
        self.automata = Automata(rule_number, num_cells)
        self.model = Rule30CNN()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epoch = 0
        self.loss = 0

    def generate_data(
        self, num_records: int = 50000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for a 1D cellular automaton.

        Parameters
        ----------
        num_records : int
            The number of training samples to generate.
        num_cells : int
            The length of each training sample.
        rule_number : int
            The rule number to use for the cellular automaton.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the input and output training data.
        """
        # Create a new automaton
        automata = Automata(self.rule_number, self.num_cells)

        data = []
        labels = []
        for _ in range(num_records):
            # Generate a random initial state
            initial_state = np.random.randint(2, size=self.num_cells)

            # Generate the next state
            next_state = automata.apply_rule(initial_state)

            data.append(initial_state)
            labels.append(next_state)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )

    def prepare_data(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training and test data for a 1D cellular automaton.

        Parameters
        ----------
        num_cells : int
            The length of each training sample.
        rule_number : int
            The rule number to use for the cellular automaton.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the training and test data and labels.
        """
        data, labels = self.generate_data()
        data = data.view(-1, 1, self.num_cells)
        labels = labels.view(-1, self.num_cells)
        split = int(0.8 * len(data))
        train_data, test_data = data[:split], data[split:]
        train_labels, test_labels = labels[:split], labels[split:]

        return train_data, test_data, train_labels, test_labels

    def early_stopping(self, actual_automata: "Automata") -> float:
        """
        Perform early stopping based on the accuracy of the model to the actual
        cellular automaton.

        Parameters
        ----------
        actual_automata : Automata
            The actual cellular automaton to compare with.

        Returns
        -------
        float
            The percentage of matches between the actual automaton and the model's
            predictions.
        """
        # Prepare a new initial state
        initial_state = [0] * self.num_cells  # Initialize with all zeros
        initial_state[self.num_cells // 2] = 1  # Set the middle cell to 1
        current_state = torch.tensor(initial_state, dtype=torch.float32).view(1, 1, -1)
        predictions = [current_state.view(-1).numpy()]

        with torch.no_grad():
            for _ in range(self.num_generations - 1):
                output = self.model(current_state)
                current_state = (output > 0.5).float()  # Binarize the output
                predictions.append(current_state.view(-1).numpy())

        match = Automata.compare(actual_automata, predictions)

        return match

    def train(self, epochs: int = 2000) -> None:
        """
        Train a model to predict the next state of a 1D cellular automaton.

        The training will stop early if the model achieves a match of 100% with
        the actual cellular automaton.

        Parameters
        ----------
        epochs : int
            The number of epochs to train the model.
        """

        real_automata = self.automata.generate(self.num_generations)

        train_data, _, train_labels, _ = self.prepare_data()

        self.epoch = 0
        while True:
            self.optimizer.zero_grad()
            output = self.model(train_data)
            loss = self.criterion(output, train_labels)
            loss.backward()
            self.optimizer.step()
            if self.epoch % 10 == 0:
                match = self.early_stopping(real_automata)
                print(f"Epoch {self.epoch}, Loss: {loss.item()}, Match: {match:.2f}%")
                if match > 99.99:
                    print(f"Early stopping at epoch {self.epoch}")
                    break
            self.epoch += 1
            self.loss = loss.item()

            if self.epoch == epochs:
                print(f"Training completed at epoch {self.epoch}")
                break
