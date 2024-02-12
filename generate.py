import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from automata import apply_ca_rule, generate_automata
from visualize import cs_plot, plot

# Parameters
rule_number = 73  # Rule number
num_cells = 101  # Number of cells in a row
num_generations = 100  # Number of generations
initial_state = [0] * num_cells  # Initialize with all zeros
initial_state[num_cells // 2] = 1  # Set the middle cell to 1

# Generate the cellular automata
automata = generate_automata(rule_number, initial_state, num_generations)

plot = cs_plot(automata, rule_number)
plt.show()


# Generate training data
def generate_data(size=10000, length=15):
    data = []
    labels = []
    for _ in range(size):
        current_gen = np.random.randint(2, size=length)
        next_gen = apply_ca_rule(current_gen, rule_number)
        data.append(current_gen)
        labels.append(next_gen)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(
        labels, dtype=torch.float32
    )


# Define a simple 1D CNN model
class Rule30CNN(nn.Module):
    def __init__(self):
        super(Rule30CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(10 * length, length)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 10 * length)
        x = torch.sigmoid(self.fc1(x))
        return x


# Prepare the data
length = 101
data, labels = generate_data(length=length)
data = data.view(-1, 1, length)
labels = labels.view(-1, length)
split = int(0.8 * len(data))
train_data, test_data = data[:split], data[split:]
train_labels, test_labels = labels[:split], labels[split:]

# Initialize the model, loss function, and optimizer
model = Rule30CNN()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1500
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

    # if epoch % 100 == 0:
    #     plot(model, test_data, test_labels, criterion, epoch)


def predict_and_visualize(model, initial_state, num_generations=100):
    current_state = torch.tensor(initial_state, dtype=torch.float32).view(1, 1, -1)
    predictions = [current_state.view(-1).numpy()]

    with torch.no_grad():
        for _ in range(num_generations - 1):
            output = model(current_state)
            current_state = (output > 0.5).float()  # Binarize the output
            predictions.append(current_state.view(-1).numpy())

    plt.figure(figsize=(10, 10))
    plt.imshow(predictions, cmap="binary", interpolation="nearest")
    plt.title("Predicted Rule 30 Cellular Automaton")
    plt.axis("off")
    plt.show()


# Prepare a new initial state
new_initial_state = [0] * num_cells  # Initialize with all zeros
new_initial_state[num_cells // 2] = 1  # Set the middle cell to 1

# Predict and visualize
predict_and_visualize(model, new_initial_state, num_generations=100)
