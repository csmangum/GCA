import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# Parameters
width = 75  # Width of the CA, must be an odd number

# Rule 30 conversion to binary representation
rule_number = 30
rule_string = np.binary_repr(rule_number, width=8)
rule = np.array([int(bit) for bit in rule_string])[::-1]

# Initialize the CA with a single row having a single 1 in the middle
initial_state = np.zeros(width, dtype=int)
initial_state[width // 2] = 1
ca = [initial_state]


# Define the update function for the CA for a single step
def update_ca(last_row):
    new_row = np.zeros_like(last_row)
    for j in range(1, len(last_row) - 1):
        pattern = 4 * last_row[j - 1] + 2 * last_row[j] + last_row[j + 1]
        new_row[j] = rule[pattern]
    return new_row


# Animation setup
fig, ax = plt.subplots()
img = ax.imshow([initial_state], cmap="binary", interpolation="nearest", aspect="auto")
ax.set_axis_off()


# Animation update function for adding new CA steps at the bottom and scrolling upwards
def update(frame):
    if frame == 0:  # Reset CA on first frame to ensure clean start on restarts
        ca.clear()
        ca.append(initial_state)
    else:
        new_row = update_ca(ca[0])
        ca.insert(0, new_row)  # Insert new row at the beginning
        if len(ca) > 50:  # Keep the display to the last 50 rows for memory efficiency
            ca.pop()  # Remove the oldest row from the end
    img.set_array(np.array(ca))
    return (img,)


# Create animation that repeats indefinitely
ani = animation.FuncAnimation(
    fig, update, frames=range(1000), interval=25, blit=True, repeat=True
)

plt.show()
