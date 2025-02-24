from typing import Dict, Tuple



def dict_to_sorted_list(dictionary: Dict[int, str]) -> list[Tuple[int, str]]:

    """

    Converts a dictionary to a sorted list of tuples based on the dictionary's keys.



    Args:

        dictionary: A dictionary where keys are IDs and values are any type.



    Returns:

        A list of tuples, where each tuple contains an ID and its corresponding value.

        The list is sorted based on the IDs.

    """

    return sorted(dictionary.items(), key=lambda x: x[0])  # Sort tuples based on the first element (ID)

