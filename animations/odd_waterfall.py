import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sympy import primerange


# Function to check if a number is prime
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def initial_state(width, number = 1, type="middle"):
    initial_state = np.zeros(width, dtype=int)
    
    if type == "middle":
        initial_state[width // 2] = 1
    elif type == "random":
        for i in range(number):
            initial_state[np.random.randint(0, width)] = 1
    elif type == "left":
        initial_state[0] = 1
    elif type == "right":
        initial_state[-1] = 1
    return initial_state


# Function to generate rule numbers based on selected option
def generate_rule_numbers(option):
    if option == "all":
        return np.arange(256)
    elif option == "odds":
        return np.arange(1, 256, 2)
    elif option == "evens":
        return np.arange(0, 256, 2)
    elif option == "primes":
        return np.array(list(primerange(0, 256)))
    elif option.isdigit():
        return [int(option)]
    else:
        raise ValueError("Invalid option")


# Parameters
width = 75  # Width of the CA, must be an odd number


# Function to initialize the rule from a rule number
def init_rule(rule_number):
    rule_string = np.binary_repr(rule_number, width=8)
    return np.array([int(bit) for bit in rule_string])[::-1]

#! do a version where the array wraps around

# User selection
rule_option = (
    "30"  # Choose from: "odds", "evens", "primes", "all", "<static rule number>"
)
order = "sequential"  # Choose from: "random", "sequential", "reverse", "super-random"


# Generate rule numbers based on the selected option
rule_numbers = generate_rule_numbers(rule_option)
if order == "random":
    np.random.seed(42)  # Optional: for reproducible results
    np.random.shuffle(rule_numbers)

if order == "reverse":
    rule_numbers = rule_numbers[::-1]

if order == "super-random":
    # random with replacement
    rule_numbers = np.random.choice(rule_numbers, size=len(rule_numbers), replace=True)

# Initialize the CA with a single row having a single 1 in the middle
# middle, random, left, right
initial_state = initial_state(width, number=15, type="middle")

# Initialize the CA state
ca = [initial_state for _ in range(50)]  # Start with 50 identical rows

# Animation setup
fig, ax = plt.subplots()
img = ax.imshow(ca, cmap="binary", interpolation="nearest", aspect="auto")
ax.set_axis_off()

# Add text for displaying the current rule number, positioned at the top left corner
rule_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")


# Update function for the CA for a single step, now including rule update and text update
def update(frame):
    global ca
    rule_number = rule_numbers[frame % len(rule_numbers)]  # Use rule from the list
    rule = init_rule(rule_number)

    new_row = np.zeros_like(ca[0])
    for j in range(1, len(ca[0]) - 1):
        pattern = 4 * ca[0][j - 1] + 2 * ca[0][j] + ca[0][j + 1]
        new_row[j] = rule[pattern]

    ca.pop()  # Remove the oldest row from the end
    ca.insert(0, new_row)  # Insert new row at the beginning
    img.set_array(np.array(ca))

    # Update the text to show the current rule number
    rule_text.set_text(f"Rule: {rule_number}")

    return (img, rule_text)


# Create animation that repeats indefinitely
ani = animation.FuncAnimation(
    fig,
    update,
    frames=np.arange(len(rule_numbers)),
    interval=25,
    blit=True,
    repeat=True,
)

plt.show()


class Waterfall:
    def __init__(self, width, rule_number):
        self.width = width
        self.rule_number = rule_number
        self.rule = self.init_rule(rule_number)
        self.initial_state = np.zeros(width, dtype=int)
        self.initial_state[width // 2] = 1
        self.ca = [self.initial_state for _ in range(50)]

    def init_rule(self, rule_number):
        rule_string = np.binary_repr(rule_number, width=8)
        return np.array([int(bit) for bit in rule_string])[::-1]

    def update_ca(self, last_row):
        new_row = np.zeros_like(last_row)
        for j in range(1, len(last_row) - 1):
            pattern = 4 * last_row[j - 1] + 2 * last_row[j] + last_row[j + 1]
            new_row[j] = self.rule[pattern]
        return new_row

    def update(self, frame):
        new_row = self.update_ca(self.ca[0])
        self.ca.insert(0, new_row)
        if len(self.ca) > 50:
            self.ca.pop()
        self.img.set_array(np.array(self.ca))
        return (self.img,)

    def generate_rule_numbers(self, option, order):
        if option == "all":
            return np.arange(256)
        elif option == "odds":
            return np.arange(1, 256, 2)
        elif option == "evens":
            return np.arange(0, 256, 2)
        elif option == "primes":
            return np.array(list(primerange(0, 256)))
        elif option.isdigit():
            return [int(option)]
        else:
            raise ValueError("Invalid option")

    def animate(self, order="sequential", rule_option="30"):
        rule_numbers = self.generate_rule_numbers(rule_option, order)
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(
            [self.initial_state], cmap="binary", interpolation="nearest", aspect="auto"
        )
        self.ax.set_axis_off()
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=np.arange(len(rule_numbers)),
            interval=25,
            blit=True,
            repeat=True,
        )
        plt.show()


# # Example usage
# waterfall = Waterfall(75, 30)
# waterfall.animate()
