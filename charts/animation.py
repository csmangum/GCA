import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.lines import Line2D


def scatter_animation(layer_1, layer_2):
    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size as needed
    num_neurons = max(len(layer_1[0]), len(layer_2[0]))

    # Initialize scatter plots for each layer
    (scatter1,) = ax.plot([], [], "o", color="blue", label="Layer 1")
    (scatter2,) = ax.plot([], [], "o", color="red", label="Layer 2")

    ax.set_xlabel("Neuron Value")
    ax.set_ylabel("Neuron Index")

    all_values = np.hstack(layer_1 + layer_2)
    max_abs_value = max(abs(all_values.min()), abs(all_values.max()))
    padding = max_abs_value * 0.05  # Additional 5% padding

    ax.set_xlim(-max_abs_value - padding, max_abs_value + padding)
    ax.set_ylim(-1, num_neurons + 1)

    ax.axvline(x=0, color="gray", linestyle="dotted", alpha=0.5)

    def init():
        scatter1.set_data([], [])
        scatter2.set_data([], [])
        return (
            scatter1,
            scatter2,
        )

    def update(frame):
        x_data1 = layer_1[frame] if frame < len(layer_1) else np.array([])
        y_data1 = range(len(x_data1))
        scatter1.set_data(x_data1, y_data1)

        x_data2 = layer_2[frame] if frame < len(layer_2) else np.array([])
        y_data2 = range(len(x_data2))
        scatter2.set_data(x_data2, y_data2)

        for line in ax.lines[:]:
            gid = line.get_gid() or ""  # Use an empty string if get_gid() returns None
            if "avg_line" in gid:
                line.remove()

        if len(x_data1) > 0:
            avg1 = np.mean(x_data1)
            ax.axvline(
                x=avg1, color="blue", linestyle="dotted", alpha=0.5, gid="avg_line_1"
            )
        if len(x_data2) > 0:
            avg2 = np.mean(x_data2)
            ax.axvline(
                x=avg2, color="red", linestyle="dotted", alpha=0.5, gid="avg_line_2"
            )

        ax.set_title(f"Epoch {frame + 1}")
        return (
            scatter1,
            scatter2,
        )

    avg_line1 = Line2D(
        [0], [0], color="blue", linestyle="dotted", lw=2, label="Avg Layer 1"
    )
    avg_line2 = Line2D(
        [0], [0], color="red", linestyle="dotted", lw=2, label="Avg Layer 2"
    )

    ax.legend(
        handles=[scatter1, scatter2, avg_line1, avg_line2],
        loc="upper left",
        bbox_to_anchor=(1, 1),
    )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max(len(layer_1), len(layer_2)),
        init_func=init,
        blit=True,
        interval=100,
    )

    fig.subplots_adjust(
        right=0.75
    )  # Adjust this value to provide enough space for the legend

    return HTML(ani.to_jshtml())
