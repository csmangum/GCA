from io import BytesIO

import imageio.v2 as imageio
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def evolution(automata: np.ndarray, rule: int) -> str:
    """
    Create a gif with the evolution of the automata, where the most recent
    generation is highlighted in red.

    Parameters
    ----------
    automata : np.ndarray
        The automata to be animated.
    rule : int
        The rule used to generate the automata.

    Returns
    -------
    str
        The path to the generated gif.
    """
    images = []
    cmap = ListedColormap(["white", "black", "red"])
    n_rows, n_cols = automata.shape

    # Pre-create figure and axes outside the loop for efficiency
    fig = Figure(figsize=(10, 20), dpi=80)
    ax = fig.subplots()
    ax.axis("off")

    # Create a buffer to reuse for saving images
    buf = BytesIO()

    for i in range(1, n_rows + 1):
        partial_automata = np.copy(automata[:i])
        overlay = np.zeros_like(partial_automata)

        if i > 1:
            overlay[-1, partial_automata[-1] == 1] = 2

        partial_automata[partial_automata == 1] = 1
        partial_automata = np.maximum(partial_automata, overlay)

        # Add empty rows to keep the frame size constant
        if i < n_rows:
            empty_rows = np.zeros((n_rows - i, n_cols))
            partial_automata = np.vstack((partial_automata, empty_rows))

        ax.clear()
        ax.imshow(partial_automata, cmap=cmap, interpolation="none", vmin=0, vmax=2)
        ax.set_title(f"Rule {rule} - Generation {i}", fontsize=20)
        ax.axis("off")

        # Reuse the buffer to save memory
        buf.seek(0)
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        buf.seek(0)
        images.append(imageio.imread(buf))

    gif_path = f"results/rule_{rule}/automata_evolution.gif"
    imageio.mimsave(gif_path, images, duration=0.1)
    buf.close()  # Close the buffer

    return gif_path
