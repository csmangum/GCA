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
        self.num_cells = num_cells
        self.initial_state = self._initial_state()

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

    def execute(
        self,
        num_generations: int,
        num_cells: int = 101,
        initial_state: np.ndarray = None,
    ) -> np.ndarray:
        """
        Execute a 2D array of cell states for a 1D cellular automaton.

        Parameters
        ----------
        num_generations : int
            The number of rows of cells to execute.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of cell states with shape (num_generations, initial_state.size).
        """

        self.num_generations = num_generations

        if initial_state is not None:
            self.initial_state = initial_state

        cells = self.initial_state
        history = [cells]
        for _ in range(self.num_generations - 1):
            cells = self.apply_rule(cells)
            history.append(cells)
        return np.array(history)

    def generate(self, num_generations: int, model) -> np.ndarray:
        """
        Generate a cellular automata from a provided model

        Parameters
        ----------
        num_generations : int
            The number of rows of cells to execute.
        model : model
            trained pytorch model

        Returns
        -------
        np.ndarray
            A 2D NumPy array of cell states with shape (num_generations,
            initial_state.size).
        """
        cells = self.initial_state
        history = [cells]
        for _ in range(num_generations - 1):
            cells = model.predict(cells)
            history.append(cells)
        return np.array(history)

    @staticmethod
    def compare(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compare two arrays and return the percentage of matches.

        Used to compare the actual automata vs the generated automata.

        Parameters
        ----------
        a : array_like
            The first array of cell states.
        b : array_like
            The second array of cell states.

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
            match_percentage = 0.0

        return match_percentage

    def plot(self, history: np.ndarray) -> None:
        """
        Plot the history of the cellular automata

        Parameters
        ----------
        history : np.ndarray
            The history of the cellular automata
        """
        import matplotlib.pyplot as plt

        height, width = history.shape
        fig_size = (width * 0.05, height * 0.05)  # These factors can be adjusted

        plt.figure(figsize=fig_size)
        plt.imshow(history, cmap="Greys", interpolation="nearest")

        # Remove axis for better visual appearance
        # plt.axis('off')

        plt.show()


#! not correct data coming out
class SequenceDataGenerator:
    def __init__(self, rule_number, sequence_length=10):
        self.rule_number = rule_number
        self.sequence_length = sequence_length
        self.rule_binary = np.array(
            [int(x) for x in np.binary_repr(self.rule_number, width=8)]
        )

    def _apply_rule(self, current_state):
        """Apply CA rule to the current state to get the next state."""
        next_state = np.zeros_like(current_state)
        for i in range(1, len(current_state) - 1):
            neighborhood = current_state[i - 1 : i + 2]
            rule_index = 7 - int("".join(neighborhood.astype(str)), 2)
            next_state[i] = self.rule_binary[rule_index]
        return next_state

    def generate_data(self, num_samples):
        """Generate data samples based on the specified CA rule."""
        # Adjust to create an array of shape (num_samples, self.sequence_length-1, 3) for inputs
        data = np.zeros((num_samples, self.sequence_length - 1, 3))
        targets = np.zeros((num_samples, 1))

        for i in range(num_samples):
            # Initialize a random initial state for the sequence
            state_sequence = np.random.randint(2, size=(self.sequence_length, 3))
            for j in range(1, self.sequence_length):
                state_sequence[j] = self._apply_rule(state_sequence[j - 1])

            data[i] = state_sequence[:-1]  # Use all but the last state as features
            targets[i] = state_sequence[
                -1, 1
            ]  # Target is the middle cell of the last state

        return data, targets

    def generate_backward_data(self, num_samples):
        """Generate data for backward prediction task."""
        # This is more complex and requires thoughtful implementation
        # For simplification, this method can be similar to generate_data but with reversed logic
        raise NotImplementedError("Backward data generation not implemented.")


# rule_number = 30  # Example CA rule
# data_generator = SequenceDataGenerator(rule_number)
# data, targets = data_generator.generate_data(num_samples=1000)

# print(f"Data shape: {data.shape}") # (num_samples, sequence_length-1, 3)
# print(f"Targets shape: {targets.shape}") # (num_samples, 1)
# print(f"Data sample:\n{data[0]}")
# print(f"Targets sample: {targets[0]}")
