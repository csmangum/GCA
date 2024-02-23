import os
import re
import shutil
import time


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
