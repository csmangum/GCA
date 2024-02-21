import os
import shutil
import time

from matplotlib import pyplot as plt

from learn import Learn
from settings import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def make_dir(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
        else:
            os.makedirs(path)
    except PermissionError as e:
        print(f"PermissionError encountered: {e}. Retrying...")
        time.sleep(1)  # Wait a bit in case the file is temporarily locked
        if os.path.exists(path):  # Check again to see if the directory still exists
            shutil.rmtree(path)
        else:
            os.makedirs(path)


def plot_automata(rule_number, automata, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    plt.title(f"Real Cellular Automata Rule {rule_number}")
    plt.axis("off")
    plt.savefig(path + f"real_automata.png")
    plt.close()


for i in range(0, 256):
    rule_number = i
    path = f"results/rule_{rule_number}/"
    make_dir(path)
    learn = Learn(
        rule_number=rule_number,
        num_cells=NUM_CELLS,
        num_generations=GENERATIONS,
        learning_rate=LEARNING_RATE,
        training_size=TRAINING_SIZE,
        epochs=EPOCHS,
        path=path,
    )
    real_automata = learn.automata.generate(100)
    plot_automata(rule_number, real_automata, path)
    learn.train()

# make_dir(PATH)
# learn = Learn(
#     rule_number=RULE_NUMBER,
#     num_cells=NUM_CELLS,
#     num_generations=GENERATIONS,
#     learning_rate=LEARNING_RATE,
#     training_size=TRAINING_SIZE,
#     epochs=EPOCHS,
#     path=PATH,
# )
# # real_automata = learn.automata.generate(100)
# # plot_automata(rule_number, real_automata, path)
# learn.train()
