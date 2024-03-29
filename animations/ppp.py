import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Parameters
width = 75  # Width of the CA, must be an odd number

# Function to initialize the rule from a rule number
def init_rule(rule_number):
    rule_string = np.binary_repr(rule_number, width=8)
    return np.array([int(bit) for bit in rule_string])[::-1]

# Initialize the CA with a single row having a single 1 in the middle
initial_state = np.zeros(width, dtype=int)
initial_state[width // 2] = 1

# Initialize the CA state
ca = [initial_state for _ in range(50)]  # Start with 50 identical rows

# Animation setup
fig, ax = plt.subplots()
img = ax.imshow(ca, cmap="binary", interpolation="nearest", aspect="auto")
ax.set_axis_off()

# Add text for displaying the current rule number, positioned at the top left corner
rule_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color="red")

# Update function for the CA for a single step, now including rule update and text update
def update(frame):
    global ca
    rule_number = frame % 256  # Cycle through rules
    rule = init_rule(rule_number)
    
    new_row = np.zeros_like(ca[0])
    for j in range(1, len(ca[0]) - 1):
        pattern = 4 * ca[0][j - 1] + 2 * ca[0][j] + ca[0][j + 1]
        new_row[j] = rule[pattern]
    
    ca.pop()  # Remove the oldest row from the end
    ca.insert(0, new_row)  # Insert new row at the beginning
    img.set_array(np.array(ca))
    
    # Update the text to show the current rule number
    rule_text.set_text(f'Rule: {rule_number}')
    
    return (img, rule_text)

# Create animation that repeats indefinitely
ani = animation.FuncAnimation(
    fig, update, frames=np.arange(256), interval=25, blit=True, repeat=True
)

plt.show()
