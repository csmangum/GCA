import json
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from automata import Automata
from models import GeneralAutomataCNN


def plot_automata(rule_number, automata, path, epoch):
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    plt.title(f"Generated Cellular Automata Rule {rule_number}")
    plt.axis("off")
    plt.savefig(f"{path}generated_automata_{epoch}.png")
    plt.close()


def save_array(automata, path, epoch):
    np.save(f"{path}automata_{epoch}.npy", automata)


def generate_from_model(model, rule_encoding, num_generations, num_cells):
    initial_state = torch.zeros(1, 1, num_cells)  # Initialize with all zeros
    initial_state[:, :, num_cells // 2] = 1  # Set the middle cell to 1
    rule_encoding = rule_encoding.unsqueeze(0)  # Add batch dimension
    current_state = torch.cat(
        (initial_state, rule_encoding), dim=1
    )  # Concatenate state and rule

    predictions = [initial_state.squeeze().numpy()]
    with torch.no_grad():
        for _ in range(num_generations - 1):
            output = model(current_state)
            current_state[:, :1, :] = (
                output > 0.5
            ).float()  # Update state; keep rule constant
            predictions.append(output.squeeze().numpy())
    return predictions


class Learn:
    def __init__(
        self,
        num_cells=101,
        num_generations=100,
        learning_rate=0.001,
        training_size=50000,
        epochs=3000,
        path=None,
        out_channels=20,
        kernel_size=3,
    ):
        # Attributes initialization
        self.num_cells = num_cells
        self.num_generations = num_generations
        self.learning_rate = learning_rate
        self.training_size = training_size
        self.epochs = epochs
        self.path = path or "./"
        self.model = GeneralAutomataCNN(num_cells, out_channels, kernel_size)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epoch = 0
        self.loss = 0
        self.loss_history = []

    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training data for the model by randomly selecting a rule and
        generating the next state.

        Returns
        -------
        torch.Tensor
            The input data for the model.
        """
        data, labels = [], []
        for _ in range(self.training_size):
            rule_number = np.random.randint(256)  # Randomly select a rule
            automata = Automata(rule_number, self.num_cells)
            initial_state = np.random.randint(2, size=self.num_cells)
            next_state = automata.apply_rule(initial_state)

            # Correctly encode the rule as a binary vector and repeat it to match the width
            rule_encoding = np.unpackbits(np.array([rule_number], dtype=np.uint8))
            rule_encoding_repeated = np.repeat(rule_encoding, self.num_cells // 8 + 1)[
                : self.num_cells
            ]

            combined_input = np.vstack(
                (initial_state, rule_encoding_repeated)
            )  # Stack state and rule
            data.append(combined_input)
            labels.append(next_state[1])
        return torch.tensor(data, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )

    def train(self) -> None:
        """
        Train the model on the generated data.
        """
        train_data, train_labels = self.generate_data()
        train_data = train_data.view(-1, 2, self.num_cells)
        train_labels = train_labels.view(-1, 1)

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(train_data)
            loss = self.criterion(output, train_labels)
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                self.loss_history.append(loss.item())
                print(f"Epoch {epoch}, Loss: {loss.item()}")

            if loss.item() < 0.01:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                break

        return self.loss_history
