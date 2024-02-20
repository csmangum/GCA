import json
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from automata import Automata
from model import Rule30CNN


def plot_automata(rule_number, automata, path, epoch):
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    plt.title(f"Generated Cellular Automata Rule {rule_number}")
    plt.axis("off")
    plt.savefig(path + f"generated_automata_{epoch}.png")
    plt.close()


def save_array(automata, path, epoch):
    np.save(path + f"automata_{epoch}.npy", automata)


def generate_from_model(model, num_generations, num_cells):
    initial_state = [0] * num_cells  # Initialize with all zeros
    initial_state[num_cells // 2] = 1  # Set the middle cell to 1
    current_state = torch.tensor(initial_state, dtype=torch.float32).view(1, 1, -1)
    predictions = [current_state.view(-1).numpy()]

    with torch.no_grad():
        for _ in range(num_generations - 1):
            output = model(current_state)
            current_state = (output > 0.5).float()  # Binarize the output
            predictions.append(current_state.view(-1).numpy())

    return predictions


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
        self,
        rule_number: int = 30,
        num_cells: int = 101,
        num_generations: int = 100,
        learning_rate: float = 0.001,
        training_size: int = 50000,
        epochs: int = 3000,
        path: str = None,
    ):
        self.rule_number = rule_number
        self.num_cells = num_cells
        self.learning_rate = learning_rate
        self.num_generations = num_generations
        self.training_size = training_size
        self.epochs = epochs
        self.path = path
        self.automata = Automata(rule_number, num_cells)
        self.model = Rule30CNN()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epoch = 0
        self.loss = 0

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        for _ in range(self.training_size):
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
        predictions = generate_from_model(
            self.model, self.num_generations, self.num_cells
        )

        match = Automata.compare(actual_automata, predictions)

        # plot_automata(self.rule_number, np.array(predictions), path, self.epoch)
        save_array(np.array(predictions), self.path, self.epoch)

        return match

    def finalize(self):

        # Plot the final generated automata
        predictions = generate_from_model(
            self.model, self.num_generations, self.num_cells
        )
        plot_automata(self.rule_number, np.array(predictions), self.path, self.epoch)

        # Save predictions as numpy array
        save_array(np.array(predictions), self.path, self.epoch)

        # Save training results as json
        with open(self.path + "training_results.json", "w") as f:
            json.dump(self.training_results, f)

        # Save metadata as json
        with open(self.path + "metadata.json", "w") as f:
            json.dump(self.metadata, f)

        # Save model
        torch.save(self.model.state_dict(), self.path + "model.pth")

        print(f"Training completed at epoch {self.epoch}")

    def train(self):
        """
        Train a model to predict the next state of a 1D cellular automaton.

        The training will stop early if the model achieves a match of 100% with
        the actual cellular automaton.

        Parameters
        ----------
        epochs : int
            The number of epochs to train the model.
        """
        self.training_results = []
        self.metadata = {
            "rule_number": self.rule_number,
            "num_cells": self.num_cells,
            "num_generations": self.num_generations,
            "epochs": self.epochs,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model": self.model.__class__.__name__,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer.__class__.__name__,
            "criterion": self.criterion.__class__.__name__,
        }
        real_automata = self.automata.generate(self.num_generations)
        save_array(real_automata, self.path, "real")

        train_data, _, train_labels, _ = self.prepare_data()

        self.epoch = 0

        while True:
            self.optimizer.zero_grad()
            output = self.model(train_data)
            loss = self.criterion(output, train_labels)
            loss.backward()
            self.optimizer.step()

            if self.epoch % 25 == 0:
                match = self.early_stopping(real_automata)
                self.training_results.append(
                    {"epoch": self.epoch, "loss": loss.item(), "match": match}
                )
                print(f"Epoch {self.epoch}, Loss: {loss.item()}, Match: {match:.2f}%")

                # if match > 99.999 and loss.item() < 0.1:
                if match > 99.999:
                    print(f"Early stopping at epoch {self.epoch}")
                    break

            self.epoch += 1
            self.loss = loss.item()

            if self.epoch == self.epochs:
                break

        self.finalize()
