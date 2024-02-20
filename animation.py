import glob
import os
import re
from io import BytesIO

import imageio.v2 as imageio
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PIL import Image, ImageSequence

rule = 75
automata = np.load(f"results/rule_{rule}/automata_real.npy")


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


# Example usage
evolution(automata, rule)


def extract_number(filename: str) -> int:
    """
    Extract the number from a filename using a regular expression.

    Parameters
    ----------
    filename : str
        The filename from which to extract the number.

    Returns
    -------
    int
        The extracted number.
    """
    match = re.search(r"automata_(\d+)\.npy", filename)
    if match:
        return int(match.group(1))
    return 0  # Return 0 or some default value if the pattern does not match


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
        ax.set_title(f"Rule {rule} - Generation {generation_number}", fontsize=20)
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


training_evolution(rule)


def create_composite_image(
    static_image_path: str,
    gif_path: str,
    output_gif_path: str,
    last_frame_duration: int = 4000,
) -> None:
    """
    Create a composite GIF by combining a static image with a GIF.

    Parameters
    ----------
    static_image_path : str
        The path to the static image.
    gif_path : str
        The path to the GIF.
    output_gif_path : str
        The path to save the composite GIF.
    last_frame_duration : int, optional
        The duration of the last frame in milliseconds, by default 4000.
    """
    # Load the static image
    static_image = Image.open(static_image_path)

    # Load the GIF
    gif = Image.open(gif_path)

    # Determine the target height and calculate the scaled dimensions
    target_height = min(static_image.height, gif.height)
    static_image_scaled = static_image.resize(
        (int(static_image.width * target_height / static_image.height), target_height),
        Image.LANCZOS,
    )
    gif_scaled_width = int(gif.width * target_height / gif.height)

    # Initialize a list to hold the composite frames
    frames = []

    # Iterate over each frame in the GIF
    for frame in ImageSequence.Iterator(gif):
        # Scale the GIF frame to match the target height
        frame_scaled = frame.resize((gif_scaled_width, target_height), Image.LANCZOS)

        # Create a composite image for the current frame
        total_width = static_image_scaled.width + frame_scaled.width
        composite_image = Image.new("RGBA", (total_width, target_height))

        # Paste the static image and the scaled GIF frame side by side
        composite_image.paste(static_image_scaled, (0, 0))
        composite_image.paste(frame_scaled, (static_image_scaled.width, 0))

        # Append the composite frame to the list
        frames.append(composite_image)

    # Set the duration for each frame (except the last one) using the original duration
    for frame in frames[:-1]:
        frame.info["duration"] = gif.info["duration"]

    # Set a longer duration for the last frame
    frames[-1].info["duration"] = last_frame_duration

    # Save the frames as a new GIF with modified frame durations
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        format="GIF",
        disposal=2,
    )


# Example usage
gif_path = f"results/rule_{rule}/training_evolution.gif"
static_path = f"results/rule_{rule}/real_automata_{rule}.png"  #! Need to remove the number at the end to make more consistent
output_path = f"results/rule_{rule}/composite.gif"
create_composite_image(static_path, gif_path, output_path)
