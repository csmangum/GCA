import os
import shutil

from matplotlib import pyplot as plt

from learn import Learn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

rule_number = 30
path = f"results/rule_{rule_number}/"


def make_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    os.makedirs(path)


def plot_automata(rule_number, automata, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    plt.title(f"Real Cellular Automata Rule {rule_number}")
    plt.axis("off")
    plt.savefig(path + f"real_automata_{rule_number}.png")
    plt.close()


for i in range(141, 256):
    rule_number = i
    path = f"results/rule_{rule_number}/"
    make_dir(path)
    learn = Learn(rule_number)
    real_automata = learn.automata.generate(100)
    plot_automata(rule_number, real_automata, path)
    learn.train(path=path)


# rule_number = 30
# path = f"results/rule_{rule_number}/"
# make_dir(path)
# learn = Learn(rule_number)
# real_automata = learn.automata.generate(100)
# plot_automata(rule_number, real_automata, path)
# learn.train(path=path)
