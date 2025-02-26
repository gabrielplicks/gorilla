from typing import Dict, Tuple, List



def unique_values_in_dict(d: Dict[str, str]) -> List[Tuple[str, str]]:

    """Returns a list of unique (key, value) pairs from a dictionary.



    If there are duplicate values in the dictionary, a ValueError is raised.



    Args:

        d: The dictionary to process.



    Raises:

        ValueError: If there are duplicate values in the dictionary.

    """

    values = set()

    pairs = []

    for key, value in d.items():

        if value in values:

            raise ValueError(f'Duplicate value found for key {key}')

        values.add(value)

        pairs.append((key, value))

    return pairs

