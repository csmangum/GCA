import numpy as np


class Automata:
    """
    A class to represent a 1D cellular automaton.

    Attributes
    ----------
    rule_number : int
        The rule number to use for the cellular automaton.
    initial_state : np.ndarray
        The initial state of the cellular automaton.
    num_cells : int
        The number of cells in each row of the cellular automaton.
    num_generations : int
        The number of generations to generate.

    Methods
    -------
    apply_rule(current_state)
        Apply a 1D cellular automaton rule to a row of cells.
    generate(num_generations)
        Generate a 2D array of cell states for a 1D cellular automaton.
    compare(a, b)
        Compare two arrays and return the percentage of matches.
    """

    def __init__(
        self,
        rule_number: int,
        num_cells: int,
    ) -> None:
        """
        Initialize a 1D cellular automaton.

        Parameters
        ----------
        rule_number : int
            The rule number to use for the cellular automaton.
        num_cells : int
            The number of cells in each row of the cellular automaton.
        """
        self.rule_number = rule_number
        self.initial_state = self._initial_state()
        self.num_cells = num_cells

    def _initial_state(self) -> np.ndarray:
        """
        Generate the initial state of the 1D cellular automaton.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of binary cell states (0 or 1).
        """
        initial_state = [0] * self.num_cells  # Initialize with all zeros
        initial_state[self.num_cells // 2] = 1  # Set the middle cell to 1

        return np.array(initial_state)

    def apply_rule(self, current_state: np.ndarray) -> np.ndarray:
        """
        Apply a 1D cellular automaton rule to a row of cells.

        Parameters
        ----------
        current_state : np.ndarray
            A 1D NumPy array of binary cell states (0 or 1).

        Returns
        -------
        np.ndarray
            A new array of cell states after applying the rule.
        """
        rule_binary = f"{self.rule_number:08b}"
        next_gen = []
        for i in range(len(current_state)):
            left = current_state[i - 1] if i > 0 else 0
            center = current_state[i]
            right = current_state[i + 1] if i < len(current_state) - 1 else 0
            neighborhood = 4 * left + 2 * center + right
            next_state = int(rule_binary[-1 - neighborhood])
            next_gen.append(next_state)
        return next_gen

    def generate(self, num_generations: int) -> np.ndarray:
        """
        Generate a 2D array of cell states for a 1D cellular automaton.

        Parameters
        ----------
        num_generations : int
            The number of rows of cells to generate.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of cell states with shape (num_generations, initial_state.size).
        """
        self.num_generations = num_generations
        cells = self.initial_state
        history = [cells]
        for _ in range(self.num_generations):
            cells = self.apply_rule(cells)
            history.append(cells)
        return np.array(history)

    @staticmethod
    def compare(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compare two arrays and return the percentage of matches.

        Parameters
        ----------
        a : array_like
            The first array.
        b : array_like
            The second array.

        Returns
        -------
        float
            The percentage of matches between the two arrays.
        """
        # Convert inputs to NumPy arrays if they are not already
        a = np.array(a)
        b = np.array(b)

        # Ensure arrays have the same shape
        if a.shape == b.shape:
            # Element-wise comparison
            matches = a == b

            # Calculate the percentage of matches
            match_percentage = (matches.sum() / a.size) * 100
        else:
            match_percentage = None

        return match_percentage
