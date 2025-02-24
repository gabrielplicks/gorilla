from typing import Dict, Tuple



def sort_dictionary_by_value(input_dict: Dict[int, int]) -> List[Tuple[int, int]]:

    """Sorts a dictionary by its values and returns a list of tuples.

    Args:

        input_dict: The input dictionary to be sorted.

    """

    sorted_tuples = []

    for key, value in input_dict.items():

        sorted_tuples.append((key, value))

    sorted_tuples.sort(key=lambda x: x[1])

    return sorted_tuples

