from typing import Dict, List, Tuple



def add_tuples_to_dict(d: Dict[str, List[int]], t: List[Tuple[str, int]]) -> None:

    """Adds each tuple's first element (key) to the dictionary and assigns its corresponding second element (value) as the key's value.

    If the key already exists in the dictionary, its value is appended to a list in the dictionary value.



    Args:

        d: The dictionary to add the tuples to.

        t: The list of tuples to add to the dictionary.

    """

    for k, v in t:

        d.setdefault(k, []).append(v)

