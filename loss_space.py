import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D

from models.simple import SimpleSequentialNetwork
from util import execute_rule

rule_number = 30
cell_states = execute_rule(rule_number)

X = torch.tensor([[i >> 2, (i >> 1) & 1, i & 1] for i in range(8)], dtype=torch.float32)
y = torch.tensor(cell_states, dtype=torch.float32).view(-1, 1)

model = SimpleSequentialNetwork()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize lists to store the weights and corresponding losses
weight_history = []
loss_history = []

# Train the model briefly to find a starting point, record the optimization path
for epoch in range(500):  # Let's increase the epochs to see a clearer optimization path
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Record the current weights and loss
    current_weights = model.net[0].weight.data.view(-1).numpy()
    weight_history.append(current_weights)
    loss_history.append(loss.item())

final_loss = loss_history[-1] if loss_history else None
print(f"Final loss: {final_loss}")

# Save the trained weights and biases
trained_weights = model.net[0].weight.data.clone()
trained_bias = model.net[0].bias.data.clone()

# Define two orthogonal directions
direction1 = torch.randn_like(trained_weights).view(-1)  # Flatten the tensor
direction2 = torch.randn_like(trained_weights).view(-1)  # Flatten the tensor

# Make direction2 orthogonal to direction1
direction2 -= (direction2.dot(direction1) / direction1.dot(direction1)) * direction1


# Function to compute loss given offsets along the directions
def compute_loss(offset1, offset2):
    direction1_reshaped = direction1.view_as(trained_weights)
    direction2_reshaped = direction2.view_as(trained_weights)
    new_weights = (
        trained_weights + offset1 * direction1_reshaped + offset2 * direction2_reshaped
    )
    model.net[0].weight.data = new_weights
    model.net[0].bias.data = trained_bias
    with torch.no_grad():
        y_pred = model(X)
        loss = criterion(y_pred, y)
    return loss.item()


# Create a grid of offsets to compute loss for each combination
offsets = np.linspace(-1, 1, 20)
offset1_mesh, offset2_mesh = np.meshgrid(
    offsets, offsets
)  # Create meshgrid for 3D plotting
loss_surface = np.array(
    [[compute_loss(offset1, offset2) for offset2 in offsets] for offset1 in offsets]
)

# Plotting the loss surface in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Use plot_surface to create a 3D surface plot
surf = ax.plot_surface(
    offset1_mesh, offset2_mesh, loss_surface, cmap="viridis", edgecolor="none"
)

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=5)

# Set labels and title
ax.set_title("3D Loss Surface Visualization")
ax.set_xlabel("Direction 1")
ax.set_ylabel("Direction 2")
ax.set_zlabel("Loss")

# Show plot
plt.show()
