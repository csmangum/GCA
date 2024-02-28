# from animations import evolution, training_evolution
# from charts import plot_automata, rule_states
# from learn import Learn
from matplotlib import pyplot as plt
import numpy as np
from learning.all_rules import Learn
from settings import *
from util import make_dir

import os

# Allow multiple OpenMP runtimes (workaround)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main() -> None:
    """
    Learn a cellular automaton and visualize the results.
    """

    # make_dir(PATH)
    learn = Learn(
        num_cells=NUM_CELLS,
        num_generations=GENERATIONS,
        learning_rate=LEARNING_RATE,
        training_size=TRAINING_SIZE,
        epochs=EPOCHS,
        path=PATH,
    )
    # real_automata = learn.automata.generate(100)
    # plot_automata(RULE_NUMBER, real_automata, PATH)
    loss_history = learn.train()
    # evolution(real_automata, RULE_NUMBER)
    # training_evolution(RULE_NUMBER)
    # rule_states(RULE_NUMBER)
    return loss_history


def plot_loss_curves(loss_curves, out_channels):
    """
    Plot loss curves for different out_channels.
    """
    plt.figure(figsize=(100, 50))
    for i, loss_curve in enumerate(loss_curves):
        plt.plot(loss_curve, label=f"out_channels={out_channels[i]}")
    plt.title("Loss Curves for Different out_channels")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    #! need to have a shared dataset for accurate comparison

    #! how much learning varibility is there on the same dataset?
    #! how much does that effect the final result?
    #! What happens if I have multiple versions of a dataset using the same logic to create it? Does that variance affect the final result?


def optimize():
    """
    Hyperparameter optimization with live plotting.
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()  # Create a figure and axis for plotting

    loss_curves = []
    LEARNING_RATE = np.logspace(-4, -1, 4)

    for lr in LEARNING_RATE:
        # Assuming the Learn class is correctly implemented
        learn = Learn(
            num_cells=NUM_CELLS,
            num_generations=GENERATIONS,
            learning_rate=lr,
            training_size=TRAINING_SIZE,
            epochs=EPOCHS,
            path=PATH,
        )
        learn.train()
        loss_history = learn.loss_history
        loss_curves.append(loss_history)

        # Clear the previous plot and plot the new data
        ax.clear()
        for i, loss_curve in enumerate(loss_curves):
            ax.plot(loss_curve, label=f"learning_rate={LEARNING_RATE[i]}")

        # Update the plot
        ax.legend()
        ax.set_title("Loss Curves for Different out_channels")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        plt.draw()
        plt.pause(0.1)  # Pause to update the plot

        print(f"learning_rate={lr} done")

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot


# If running as main, call the optimize function
if __name__ == "__main__":
    optimize()
