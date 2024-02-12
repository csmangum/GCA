import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt


def cs_plot(automata: np.ndarray, rule_number: int) -> plt:
    """
    Plot a 2D array of cell states as a grayscale image.

    Parameters
    ----------
    automata : np.ndarray
        A 2D NumPy array of binary cell states (0 or 1).
    rule_number : int
        An integer between 0 and 255 representing the CA rule.

    Returns
    -------
    plt
        A Matplotlib plot object.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    plt.title(f"Cellular Automata Rule {rule_number}")
    plt.axis("off")

    return plt


def plot(model, test_data, test_labels, criterion, epoch):
    with torch.no_grad():
        test_output = model(test_data)
        test_loss = criterion(test_output, test_labels)

    # Visualize the model's predictions
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Epoch: {epoch}, Test Loss: {test_loss.item()}")
    plt.subplot(1, 2, 1)
    plt.imshow(test_labels[:25].numpy(), cmap="binary", interpolation="nearest")
    plt.title("True")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(test_output[:25].numpy(), cmap="binary", interpolation="nearest")
    plt.title("Predicted")
    plt.axis("off")

    plt.savefig(f"rule30_epoch_{epoch}.png")


# Evaluation
# with torch.no_grad():
#     test_output = model(test_data)
#     test_loss = criterion(test_output, test_labels)
#     print(f"Test Loss: {test_loss.item()}")

# # Visualize the model's predictions
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(test_labels[:25].numpy(), cmap="binary", interpolation="nearest")
# plt.title("True")
# plt.axis("off")
# plt.subplot(1, 2, 2)
# plt.imshow(test_output[:25].numpy(), cmap="binary", interpolation="nearest")
# plt.title("Predicted")
# plt.axis("off")
# plt.show()

# make gif

# images = []

# matching_files = [
#     f for f in os.listdir(".") if f.startswith("rule30_epoch") and f.endswith(".png")
# ]
# matching_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
# for filename in matching_files:
#     images.append(imageio.imread(filename))
# imageio.mimsave("rule30_training.gif", images, duration=0.5)
# for filename in os.listdir("."):
#     if filename.endswith(".png"):
#         os.remove(filename)
