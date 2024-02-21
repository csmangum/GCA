import re


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
