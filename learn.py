import json
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from automata import Automata
from models.simple import SimpleSequentialNetwork


def plot_automata(
    rule_number: int, automata: np.ndarray, path: str, epoch: int
) -> None:
    """
    Plot a 2D array representing a cellular automaton and save it to a file.

    Parameters
    ----------
    rule_number : int
        The rule number of the cellular automaton.
    automata : np.ndarray
        The array representing the cellular automaton.
    path : str
        The path to save the plot to.
    epoch : int
        The epoch number of the training.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    plt.title(f"Generated Cellular Automata Rule {rule_number}")
    plt.axis("off")
    plt.savefig(path + f"generated_automata_{epoch}.png")
    plt.close()


def save_array(automata: np.ndarray, path: str, epoch: int) -> None:
    """
    Save a 2D array representing a cellular automaton to a file.

    Parameters
    ----------
    automata : np.ndarray
        The array representing the cellular automaton.
    path : str
        The path to save the array to.
    epoch : int
        The epoch number of the training.
    """
    np.save(path + f"automata_{epoch}.npy", automata)


def generate_from_model(model: nn.Module, num_generations: int, num_cells: int) -> list:
    """
    Generate a 1D cellular automaton from a machine learning model.

    Parameters
    ----------
    model : nn.Module
        The machine learning model to use for generating the cellular automaton.
    num_generations : int
        The number of generations to generate.
    num_cells : int
        The number of cells in each row of the cellular automaton.

    Returns
    -------
    List[np.ndarray]
        A list of arrays representing each generation of the cellular automaton.
    """
    initial_state = torch.zeros(1, 1, num_cells)  # Initialize with all zeros
    initial_state[:, :, num_cells // 2] = 1  # Set the middle cell to 1
    current_state = initial_state

    predictions = [initial_state.squeeze().numpy()]
    with torch.no_grad():
        for _ in range(num_generations - 1):
            output = model(current_state)
            current_state[:, :, :] = (output > 0.5).float()  # Update state
            predictions.append(output.squeeze().numpy())
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
        self.model = SimpleSequentialNetwork()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epoch = 0
        self.loss = 0
        
    def test_generate(self, num_generations, num_cells):
        """
        Generate a 1D cellular automaton from a machine learning model.
        
        Parameters
        ----------
        num_generations : int
            The number of generations to generate.
        num_cells : int
            The number of cells in each row of the cellular automaton.
        """
        initial_state = torch.zeros( 1, num_cells)  # Initialize with all zeros
        initial_state[:, num_cells // 2] = 1  # Set the middle cell to 1
        current_state = initial_state
        
        predictions = [initial_state.squeeze().numpy()]
        with torch.no_grad():
            for _ in range(num_generations - 1):
                output = self.model(current_state[0])
                current_state = (output > 0.5).float()
                predictions.append(output.squeeze().numpy())
        return predictions
    
    

        

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
            
        center_states = [[row[len(row) // 2]] for row in labels]
        
        # save data and labels to txt files
        # np.savetxt(self.path + "data.txt", data)
        # np.savetxt(self.path + "labels.txt", labels)
        
        
        
        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            center_states, dtype=torch.float32
        )

    def prepare_data(
        self,
    ):
        """
        Prepare training and test data for a 1D cellular automaton.
        
        Returns process data

        Parameters
        ----------
        num_cells : int
            The length of each training sample.
        rule_number : int
            The rule number to use for the cellular automaton.

        Returns
        -------
        
        """
        data, labels = self.generate_data()
        print(data.shape, labels.shape)
        print('----------------')
        print(data[:5])
        print('*******************')
        print(labels[:5])
        print('----------------')
        return data, labels
        

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

    def finalize(self) -> None:
        """
        Finalize the training process by saving the model and training results.
        """

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

    def train(self) -> None:
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

        data, labels = self.prepare_data()

        self.epoch = 0

        while True:
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            if self.epoch % 25 == 0:
                match = self.early_stopping(real_automata)
                self.training_results.append(
                    {"epoch": self.epoch, "loss": loss.item(), "match": match}
                )
                print(f"Epoch {self.epoch}, Loss: {loss.item()}, Match: {match:.2f}%")

                if match > 99.999 and loss.item() < 0.05:
                    print(f"Early stopping at epoch {self.epoch}")
                    break

            self.epoch += 1
            self.loss = loss.item()

            if self.epoch == self.epochs:
                break

        # self.finalize()
