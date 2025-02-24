from collections import defaultdict

from typing import Dict, List



def string_indices(string_list: List[str]) -> Dict[str, List[int]]:

    """Creates a dictionary where each key represents a unique string, and the corresponding value is a list of string indices where that string occurs.



    Args:

        string_list: A list of strings.



    Returns:

        A dictionary where each key represents a unique string, and the corresponding value is a list of string indices where that string occurs.

    """

    indices: Dict[str, List[int]] = defaultdict(list)

    for i, s in enumerate(string_list):

        indices[s].append(i)

    return indices

