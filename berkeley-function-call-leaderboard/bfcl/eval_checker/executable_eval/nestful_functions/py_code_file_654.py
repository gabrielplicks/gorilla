from typing import List, Dict



def name_lengths(names: List[str]) -> Dict[str, int]:

    """Creates a dictionary with each name as a key and the length of each name as its value.



    Args:

        names: A list of names.



    Returns:

        A dictionary with each name as a key and the length of each name as its value.

    """

    return {name: len(name) for name in names}

