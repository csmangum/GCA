import glob
import os
import re
from io import BytesIO

import imageio.v2 as imageio
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from util import extract_number


def training_evolution(rule: int) -> str:
    """
    Create a GIF of the training evolution of the generated automata.

    Loads the numpy arrays of the training evolution and creates a GIF, loading each
    array and saving it as a frame in the GIF.

    Parameters
    ----------
    rule : int
        The rule used to generate the automata.

    Returns
    -------
    str
        The path to the generated GIF.
    """

    # Identify the npy files in the rule folder
    rule_folder = f"results/rule_{rule}"
    automata_files = sorted(
        [
            os.path.basename(file)
            for file in glob.glob(f"{rule_folder}/automata_*.npy")
            if re.match(r"automata_\d+\.npy", os.path.basename(file))
        ],
        key=extract_number,  # Use the custom sort key
    )

    images = []
    cmap = ListedColormap(["white", "black"])

    # Pre-create figure and axes outside the loop for efficiency
    fig = Figure(figsize=(10, 20), dpi=80)
    ax = fig.subplots()
    ax.axis("off")

    # Create a buffer to reuse for saving images
    buf = BytesIO()

    for i, automata_file in enumerate(automata_files):
        generation_number = extract_number(automata_file)
        automata = np.load(os.path.join(rule_folder, automata_file))
        ax.clear()
        ax.imshow(automata, cmap=cmap, interpolation="none", vmin=0, vmax=1)
        ax.set_title(f"Learning Rule {rule} - Epoch {generation_number}", fontsize=20)
        ax.axis("off")

        # Reuse the buffer to save memory
        buf.seek(0)
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.2)
        buf.seek(0)
        images.append(imageio.imread(buf))

    gif_path = f"results/rule_{rule}/training_evolution.gif"
    imageio.mimsave(gif_path, images, duration=0.2)
    buf.close()  # Close the buffer

    return gif_path
