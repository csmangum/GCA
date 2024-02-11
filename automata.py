import numpy as np


def apply_ca_rule(cells: np.ndarray, rule_number: int) -> np.ndarray:
    """
    Apply a 1D cellular automaton rule to a row of cells.

    Parameters
    ----------
    cells : np.ndarray
        A 1D NumPy array of binary cell states (0 or 1).
    rule_number : int
        An integer between 0 and 255 representing the CA rule.

    Returns
    -------
    np.ndarray
        A new array of cell states after applying the rule.

    Example
    -------
    >>> apply_ca_rule(np.array([1, 0, 1]), 90)
    array([0, 1, 0])
    """
    rule_binary = f"{rule_number:08b}"
    next_gen = []
    for i in range(len(cells)):
        left = cells[i - 1] if i > 0 else 0
        center = cells[i]
        right = cells[i + 1] if i < len(cells) - 1 else 0
        neighborhood = 4 * left + 2 * center + right
        next_state = int(rule_binary[-1 - neighborhood])
        next_gen.append(next_state)
    return next_gen


def generate_automata(
    rule_number: int, initial_state: np.ndarray, num_generations: int
) -> np.ndarray:
    """
    Generate a 2D array of cell states for a 1D cellular automaton.

    Parameters
    ----------
    rule_number : int
        An integer between 0 and 255 representing the CA rule.
    initial_state : np.ndarray
        A 1D NumPy array of binary cell states (0 or 1).
    num_generations : int
        The number of rows of cells to generate.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of cell states with shape (num_generations, initial_state.size).
    """
    generations = [initial_state]
    for _ in range(num_generations - 1):
        next_gen = apply_ca_rule(generations[-1], rule_number)
        generations.append(next_gen)
    return np.array(generations)
