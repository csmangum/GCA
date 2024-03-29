from automata import Automata
import numpy as np

automata = Automata(rule_number=30, num_cells=33)


random_initial_state = np.random.randint(2, size=33)
automata._initial_state = random_initial_state
print(f"Initial state: {random_initial_state}")
print("-------------------------")
# [0 1 0]

# Generate a 2D array of cell states for a 1D cellular automaton
num_generations = 70
cell_states = automata.execute(num_generations, random_initial_state)
print(f"Cell states: {cell_states}")
# [[0 1 0]
#  [1 1 1]
#  [1 0 0]]
print("-------------------------")
# print the center values of the cell_states
print(f"Center values: {cell_states[:, len(cell_states[0]) // 2]}")

center_values = cell_states[:, len(cell_states[0]) // 2]

def death_time(center_values):
    for i in range(len(center_values) - 1, 0, -1):
        if center_values[i] != center_values[i - 1]:
            return i
    return 0
    
print(f"Died: {death_time(center_values)}")
print(f"life: {center_values[:death_time(center_values)+1]}")

print('*******************************************')

#! which initials states keep that automata more dynamic? Changes often, has "life" to it

import numpy as np
import matplotlib.pyplot as plt


def check_for_death(initial_state, rule_number, steps):
    """
    Check when the cellular automaton reaches a static state where is stays the same forever.
    
    Returns the number of steps it takes to reach a static state.
    """
    automata = Automata(rule_number, len(initial_state))
    current_state = initial_state
    for step in range(steps):
        next_state = automata.apply_rule(current_state)
        if np.all(next_state == current_state):
            return step
        current_state = next_state
    return steps
    


def calculate_activity(initial_state, rule_number, steps):
    """
    Calculate the activity level of a 1D cellular automaton.

    Parameters
    ----------
    initial_state : np.ndarray
        The initial state of the cellular automaton.
    rule_number : int
        The rule number to use for the cellular automaton.
    steps : int
        The number of time steps to simulate.

    Returns
    -------
    int
        The number of oscillations or changes in the cellular automaton state.
    """
    automata_b = Automata(rule_number, len(initial_state))
    current_state = initial_state
    activity = 0
    for _ in range(steps):
        next_state = automata_b.apply_rule(current_state)
        activity += np.sum(next_state != current_state)
        current_state = next_state
        
    death = death_time(current_state)

    print(
        f"Rule: {rule_number} Activity: {activity}, Initial State: {initial_state}, Steps: {steps}"
    )
    return activity


# Example usage
rule_number = 30  # Example CA rule
num_states = 100  # Number of initial states to test
steps = 70  # Number of steps to simulate
state_size = 7  # Size of the CA state

from itertools import product

def generate_permutations(n):
    # Generate all combinations of 0s and 1s for a list of size n
    return list(product([0, 1], repeat=n))


possible_states = generate_permutations(state_size)

all_results = []
death_times = []

for state in possible_states:
    automata_b = Automata(rule_number=30, num_cells=state_size)
    results = automata_b.execute(steps, state)
    all_results.append(results)
    # print(f"Initial state: {state}")
    # print(f"Center values: {results[:, len(results[0]) // 2]}")
    # print(f"Results: {results}")
    center_values = results[:, len(results[0]) // 2]
    
    death = death_time(center_values)
    death_times.append(death)
    
print(f"Death times: {death_times}")
print(f"Max death time: {max(death_times)}, position: {death_times.index(max(death_times))}")
    

# Plot the durations
# plt.figure(figsize=(10, 5))
# plt.bar(
#     [str(state) for state in possible_states],
#     duration,
#     color=["blue" if state in dynamic_states else "red" for state in possible_states],
# )
# plt.title("Duration of Activity for Different Initial States")
# plt.xlabel("Initial State")
# plt.ylabel("Duration")
# plt.xticks(rotation=45)
# plt.legend(["Dynamic", "Static"])
# plt.show()


