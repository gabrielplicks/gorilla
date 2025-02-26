from typing import List



def compute_average_length(strings: List[str]) -> float:

    """Calculates the average length of a list of strings.



    Args:

        strings: A list of strings.



    Returns:

        The average length of the strings.

    """

    lengths = [len(string) for string in strings]

    return sum(lengths) / len(lengths)

