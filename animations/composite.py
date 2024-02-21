from PIL import Image, ImageSequence


def composite(
    static_image_path: str,
    gif_path: str,
    output_gif_path: str,
    last_frame_duration: int = 4000,
) -> str:
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

    Returns
    -------
    str
        The path to the generated GIF.
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

    return output_gif_path


# Example usage
# gif_path = f"results/rule_{rule}/training_evolution.gif"
# static_path = f"results/rule_{rule}/real_automata.png"
# output_path = f"results/rule_{rule}/composite.gif"
# composite(static_path, gif_path, output_path)
