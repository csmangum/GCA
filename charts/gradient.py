import matplotlib.pyplot as plt
import pandas as pd


def gradient_w_loss(
    gradient: list, loss: list, rolling_mean: int = None, title: str = None
) -> None:
    """
    Plot the layer gradients and loss.

    Parameters
    ----------
    gradient : list
        The layer gradients.
    loss : list
        The loss values.
    rolling_mean : int
        The window size for the rolling mean. Default is None.
    """

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gradient", color=color)
    ax1.plot(gradient, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Loss", color=color)
    ax2.plot(loss, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    # Add gradient rolling mean
    if rolling_mean:
        rolling_mean_calc = pd.Series(gradient).rolling(window=rolling_mean).mean()
        ax1.plot(rolling_mean_calc, color="tab:green")

    if title:
        plt.title(title)
    else:
        plt.title("Layer Gradients and Loss")

    fig.tight_layout()
    plt.show()
