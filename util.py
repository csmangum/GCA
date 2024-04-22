import hashlib
import os
import re
import shutil
import time

import numpy as np
import torch
import torch.nn as nn

from automata import Automata


def model_hash(model: nn.Module) -> str:
    """
    Generate a hash of the model parameters.

    Parameters
    ----------
    model : nn.Module
        The model to hash.

    Returns
    -------
    str
        The hash of the model parameters.
    """
    hash_md5 = hashlib.md5()
    extracted_parameters = extract_parameters(model)
    hash_md5.update(extracted_parameters.tobytes())
    return hash_md5.hexdigest()


def extract_number(filename: str) -> int:
    """
    Extract the number from a filename using a regular expression.

    Parameters
    ----------
    filename : str
        The filename from which to extract the number.

    Returns
    -------
    int
        The extracted number.
    """
    match = re.search(r"automata_(\d+)\.npy", filename)
    if match:
        return int(match.group(1))
    return 0  # Return 0 or some default value if the pattern does not match


def make_dir(path: str) -> None:
    """
    Create a directory if it does not exist, and delete it if it does.

    Parameters
    ----------
    path : str
        The path to the directory to create or delete.
    """
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


def early_stopping(
    real_automata: Automata, generated_automata: Automata, generations: int
) -> float:
    """
    Quantifies the difference between the real automata and the generated automata

    Parameters
    ----------
    real_automata : Automata
        The real automata
    generated_automata : Automata
        The generated automata
    generations : int
        The number of generations to generate the automata

    Returns
    -------
    float
        The difference between the two automaton
    """
    learned_results = real_automata.generate(generations, generated_automata)
    return Automata.compare(real_automata, learned_results)


def extract_parameters(model: nn.Module) -> np.ndarray:
    """
    Extract the parameters from a model. (It's weights and biases)

    Returns a flattened list of parameters. First the weights, then the biases.
    Then flattens that into a single list.

    Parameters
    ----------
    model : nn.Module
        The model from which to extract the parameters.

    Returns
    -------
    np.ndarray
        The extracted parameters in a single list.
    """
    parameters = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            parameters.append(layer.weight.data.numpy().flatten())
            parameters.append(layer.bias.data.numpy().flatten())
    return np.concatenate(parameters)


def save_model(
    tags: str,
    model: nn.Module,
    filename: str,
    parameter_history: list,
    loss_history: list,
    run_name: str,
    rule_number: int,
):
    """
    Save the model to a file

    Parameters
    ----------
    tags : str
        The tags to save with the model
    model : nn.Module
        The model to save
    filename : str
        The filename to save the model to
    parameter_history : list
        The history of the parameters
    loss_history : list
        The history of the loss
    run_name : str
        The name of the run
    rule_number : int
        The rule number of the automata
    """

    if not os.path.exists(f"history/{run_name}"):
        os.makedirs(f"history/{run_name}")

    model_history = {
        "tag": tags,
        "loss_history": loss_history,
        "weight_history": parameter_history,
        "epoch_count": len(parameter_history) - 1,
        "state_dict": model.state_dict(),
        "metadata": {
            "architecture": "SimpleSequentialNetwork",
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "learning_rate": 0.01,
            "rule_number": rule_number,
        },
    }

    torch.save(model_history, f"history/{run_name}/{filename}.pt")


def execute_rule(rule_number: int) -> list:
    """
    Execute the rule for every possible input state and
    return the output states

    Parameters
    ----------
    rule_number : int
        The rule number to be executed

    Returns
    -------
    list
        The outputs of the rule, for example, [0, 1, 1, 0, 1, 0, 0, 1]
    """
    binary_rule = np.binary_repr(rule_number, width=8)
    outputs = [int(bit) for bit in binary_rule[::-1]]
    return outputs
