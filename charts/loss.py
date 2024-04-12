import matplotlib.pyplot as plt
import numpy as np


def find_elbow_point(y_values: list) -> int:
    """
    Find the elbow point in the loss curve.

    The elbow point is the point where the loss curve starts to flatten out.

    It is calculated using the distance of each point to the line connecting
    the first and last points.

    Parameters
    ----------
    y_values : list
        The loss values.

    Returns
    -------
    int
        The index of the elbow point.
    """
    # Ensure y_values is a NumPy array for vectorized operations
    y_values = np.array(y_values)

    # Normalize the data
    min_y = np.min(y_values)
    max_y = np.max(y_values)
    norm_y = (y_values - min_y) / (max_y - min_y)

    # Line vector from first to last point
    line_vec = np.array(
        [len(norm_y) - 1, norm_y[-1] - norm_y[0]]
    )  # end point - start point

    # Normalize the line vector
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Find the elbow
    distances = []
    for i in range(len(norm_y)):
        point_vec = np.array([i, norm_y[i] - norm_y[0]])

        # Calculate the distance of the point to the line
        dist_to_line = np.abs(np.cross(line_vec_norm, point_vec)) / np.linalg.norm(
            line_vec
        )
        distances.append(dist_to_line)

    elbow_point = np.argmax(distances)
    return elbow_point


def loss_w_elbow(losses: list) -> None:
    """
    Plot the loss curve with the elbow point.

    The elbow point is the point where the loss curve starts to flatten out.

    Parameters
    ----------
    losses : list
        The loss values.
    """

    # Finding the elbow point
    elbow_index = find_elbow_point(losses)

    # Plot the curve with the elbow point
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Loss over epochs")
    plt.scatter(elbow_index, losses[elbow_index], color="red", label="Elbow point")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
