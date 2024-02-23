import os

import matplotlib.pyplot as plt
import torch

from automata import Automata
from models import AutomataCNN
from settings import *

rule_number = RULE_NUMBER
generations = EVAL_GENERATIONS

# Generate the real automata
automata = Automata(rule_number, NUM_CELLS)
states = automata.generate(generations)

plt.figure(figsize=(10, 50))
plt.imshow(states, cmap="binary", interpolation="nearest")
plt.title(f"Real", fontsize=40)
plt.axis("off")
plt.savefig(
    f"results/rule_{rule_number}/extended_real_automata.png", bbox_inches="tight"
)
plt.close()

# Load the model from file
model = AutomataCNN()
model.load_state_dict(torch.load(f"results/rule_{rule_number}/model.pth"))


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


# Generate the automata using the model
predictions = generate_from_model(model, generations, 101)

plt.figure(figsize=(10, 50))
plt.imshow(predictions, cmap="binary", interpolation="nearest")
plt.title(f"Generated", fontsize=40)
plt.axis("off")
plt.savefig(
    f"results/rule_{rule_number}/extended_generated_automata.png", bbox_inches="tight"
)
plt.close()

print(f"Real automata shape: {states.shape}")
print(f"Predicted automata shape: {len(predictions[0])}")

# Compare the predictions to the actual automata
accuracy = automata.compare(states, predictions)
print(f"Accuracy: {accuracy:.2%}")


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

A = np.array(states)
B = np.array(predictions)

# Create a color-coded array based on B's values (0 = white, 1 = black)
# Initialize with ones, which will correspond to white (to be defined in the colormap)
color_coded = np.ones(B.shape)

# Apply black color coding for values of 1 in B
color_coded[B == 1] = 0

# Identify differences and mark them with 2 (to be defined as red in the colormap)
color_coded[A != B] = 2

# Define a custom colormap: 0 = black, 1 = white, 2 = red
cmap = mcolors.ListedColormap(["black", "white", "red"])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plot the color-coded array
plt.figure(figsize=(10, 10))
plt.imshow(color_coded, cmap=cmap, norm=norm, interpolation="nearest")

plt.axis("off")

# Show plot
plt.show()
