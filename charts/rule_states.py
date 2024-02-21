import matplotlib.patches as patches
import matplotlib.pyplot as plt


def rule_states(rule_number: int) -> None:
    """
    Visualize a 1D cellular automaton rule states and their transitions.

    Parameters
    ----------
    rule_number : int
        The rule number for the 1D cellular automaton.
    """
    rule_binary = format(rule_number, "08b")[
        ::-1
    ]  # Rule binary representation, reversed for indexing

    fig, axs = plt.subplots(1, 8, figsize=(24, 2.5))  # Figure size adjusted for clarity

    padding_between_states = (
        0.5  # Padding between the current state and the new cell state
    )

    # Adjust the centering by shifting cells slightly to the left
    shift_left_amount = 0.5  # Amount to shift the cells left for better centering

    for i, ax in enumerate(axs):
        state_str = format(i, "03b")  # Current state as a binary string

        # Adjusted center_x for slight leftward shift
        center_x = (4 - 3) / 2 - shift_left_amount

        # Draw a white box with a black border for visual separation
        ax.add_patch(
            patches.Rectangle(
                (-1, 0),
                5,
                3 + padding_between_states,
                linewidth=2,
                edgecolor="black",
                facecolor="white",
            )
        )

        # Drawing the current cell state and its neighbors, centered with leftward adjustment
        for pos, bit in enumerate(state_str):
            cell_color = "black" if bit == "1" else "white"
            ax.add_patch(
                patches.Rectangle(
                    (center_x + pos, 2),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=cell_color,
                )
            )

        # Drawing the new cell state with padding, centered beneath the current state with leftward adjustment
        result_state = rule_binary[i]
        result_color = "black" if result_state == "1" else "white"
        ax.add_patch(
            patches.Rectangle(
                (center_x + 1, 1 - padding_between_states / 2),
                1,
                1,
                linewidth=1,
                edgecolor="black",
                facecolor=result_color,
            )
        )

        # Adjusting axes and aspect ratio for visual clarity and centering
        ax.set_xlim(-1, 4)
        ax.set_ylim(0, 3 + padding_between_states)
        ax.axis("off")
        ax.set_aspect("equal")  # Ensure the cells are square

    # Add a title below the figure
    plt.suptitle(f"Cellular Automaton Rule {rule_number}", fontsize=16, y=0.18)

    plt.tight_layout()
    plt.show()


# Visualize Rule 30 with cells slightly shifted to the left
rule_states(73)
