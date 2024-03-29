import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class Waterfall:
    def __init__(self, width, rule_number):
        self.width = width
        self.rule_number = rule_number
        self.rule = np.binary_repr(rule_number, width=8)
        
        # Initialize the CA with a single row having a single 1 in the middle
        self.initial_state = np.zeros(width, dtype=int)
        self.initial_state[width // 2] = 1
        self.ca = [self.initial_state]
        
        # add 50 padding row of zeros
        for i in range(50):
            self.ca.append(np.zeros(width, dtype=int))
            
    # Define the update function for the CA for a single step
    def update_ca(self, last_row):
        new_row = np.zeros_like(last_row)
        for j in range(1, len(last_row) - 1):
            pattern = 4 * last_row[j - 1] + 2 * last_row[j] + last_row[j + 1]
            new_row[j] = self.rule[pattern]
        return new_row
    
    def setup_plot(self):
        fig, ax = plt.subplots()
        img = ax.imshow([self.initial_state], cmap="binary", interpolation="nearest", aspect="auto")
        ax.set_axis_off()
        return fig, ax, img
    
    def update(self, frame):
        
        new_row = self.update_ca(self.ca[0])
        self.ca.insert(0, new_row)
        if len(self.ca) > 50:
            self.ca.pop()
        self.img.set_array(np.array(self.ca))
        return (self.img,)
    
    
    def animate(self):
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow([self.initial_state], cmap="binary", interpolation="nearest", aspect="auto")
        self.ax.set_axis_off()
        # Create animation that repeats indefinitely
        ani = animation.FuncAnimation(self.fig, self.update, frames=range(1000), interval=25, blit=True, repeat=True)

        plt.show()
        
# Example usage
# waterfall = Waterfall(75, 30)
# waterfall.animate()

# Parameters
width = 75  # Width of the CA, must be an odd number

# Rule 30 conversion to binary representation
rule_number = 86
rule_string = np.binary_repr(rule_number, width=8)
rule = np.array([int(bit) for bit in rule_string])[::-1]

# Initialize the CA with a single row having a single 1 in the middle
initial_state = np.zeros(width, dtype=int)
initial_state[width // 2] = 1
ca = [initial_state]

# add 50 padding row of zeros
for i in range(50):
    ca.append(np.zeros(width, dtype=int))


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
