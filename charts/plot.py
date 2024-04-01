import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from matplotlib import pyplot as plt


def infer_title(rule_number: int, epoch: int = None) -> str:
    """
    Infer the title of the plot.

    Parameters
    ----------
    rule_number : int
        The rule number of the cellular automata.
    epoch : int
        The epoch number of the cellular automata.

    Returns
    -------
    str
        The title of the plot.
    """
    if epoch:
        return f"Cellular Automata Rule {rule_number} - Epoch {epoch}"
    return f"Cellular Automata Rule {rule_number}"


def infer_path(path: str, rule_number: int, epoch: int = None) -> str:
    """
    Infer the path to save the plot.

    Parameters
    ----------
    path : str
        The path to save the plot.
    rule_number : int
        The rule number of the cellular automata.
    epoch : int
        The epoch number of the cellular automata.

    Returns
    -------
    str
        The path to save the plot.
    """
    if epoch:
        return path + f"predicted_automata_epoch_{epoch}.png"
    return path + f"real_automata_{rule_number}.png"


def plot_automata(
    rule_number: int,
    automata: list,
    path: str,
    title: str = None,
    epoch: int = None,
    save: bool = False,
    show: bool = True,
) -> None:
    """
    Plot the cellular automata.

    Parameters
    ----------
    rule_number : int
        The rule number of the cellular automata.
    automata : list
        The cellular automata.
    path : str
        The path to save the plot.
    title : str
        The title of the plot.
    epoch : int
        The epoch number of the cellular automata.
    save : bool
        Whether to save the plot or not.
    show : bool
        Whether to show the plot or not.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    if not title:
        title = infer_title(rule_number, epoch)
    plt.title(title, fontsize=20)
    plt.axis("off")
    if save:
        if not path:
            path = infer_path(path, rule_number, epoch)
        else:
            path = path + f"real_automata.png"
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def plot_loss(
    loss: list, path: str = None, title: str = None, show: bool = True
) -> None:
    """
    Plot the loss history

    Parameters
    ----------
    loss : list
        The loss history of the model.
    path : str
        The path to save the plot.
    title : str
        The title of the plot.
    show : bool
        Whether to show the plot or not.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    if title:
        plt.title(title, fontsize=20)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    if path:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def plot_3d_scatter(encoded_weights: np.ndarray, loss_history: np.ndarray) -> None:
    """
    Create a 3D scatter plot using Plotly.

    Parameters
    ----------
    encoded_weights : np.ndarray
        The encoded weights of the autoencoder.
    loss_history : np.ndarray
        The loss history of the autoencoder.
    """
    # Create a 3D scatter plot using Plotly
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=encoded_weights[:, 0],  # X coordinates
                y=encoded_weights[:, 1],  # Y coordinates
                z=encoded_weights[:, 2],  # Z coordinates
                mode="markers",
                marker=dict(
                    size=5,
                    color=loss_history,  # Set color to an array/list of desired values
                    colorscale="RdYlGn",  # Choose a color scale
                    colorbar=dict(title="Loss"),  # Add colorbar title
                    opacity=0.8,
                ),
            )
        ]
    )

    # Customize the layout of the plot
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),  # Reduce plot margins
        scene=dict(
            xaxis_title="X Axis Title",  # Customize X axis title
            yaxis_title="Y Axis Title",  # Customize Y axis title
            zaxis_title="Z Axis Title",  # Customize Z axis title
            aspectmode="cube",  # Equalize the axes
        ),
    )

    # Show the plot
    fig.show()
