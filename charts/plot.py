from matplotlib import pyplot as plt


def infer_title(rule_number: int, epoch: int = None) -> str:
    if epoch:
        return f"Cellular Automata Rule {rule_number} - Epoch {epoch}"
    return f"Cellular Automata Rule {rule_number}"


def infer_path(path: str, rule_number: int, epoch: int = None) -> str:
    if epoch:
        return path + f"predicted_automata_epoch_{epoch}.png"
    return path + f"real_automata_{rule_number}.png"


def plot_automata(
    rule_number: int,
    automata: list,
    path: str,
    title: str = None,
    epoch: int = None,
    save: bool = False,
    show: bool = True,
):
    plt.figure(figsize=(10, 10))
    plt.imshow(automata, cmap="binary", interpolation="nearest")
    if not title:
        title = infer_title(rule_number, epoch)
    plt.title(title, fontsize=20)
    plt.axis("off")
    if save:
        if not path:
            path = infer_path(path, rule_number, epoch)
        else:
            path = path + f"real_automata_{rule_number}.png"
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()
