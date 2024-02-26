# from animations import evolution, training_evolution
# from charts import plot_automata, rule_states
# from learn import Learn
from learning.all_rules import Learn
from settings import *
from util import make_dir


def main() -> None:
    """
    Learn a cellular automaton and visualize the results.
    """

    # make_dir(PATH)
    learn = Learn(
        num_cells=NUM_CELLS,
        num_generations=GENERATIONS,
        learning_rate=LEARNING_RATE,
        training_size=TRAINING_SIZE,
        epochs=EPOCHS,
        path=PATH,
    )
    # real_automata = learn.automata.generate(100)
    # plot_automata(RULE_NUMBER, real_automata, PATH)
    learn.train()
    # evolution(real_automata, RULE_NUMBER)
    # training_evolution(RULE_NUMBER)
    # rule_states(RULE_NUMBER)


if __name__ == "__main__":
    main()
