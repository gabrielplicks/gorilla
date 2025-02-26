from typing import Dict, Tuple



def convert_dict_to_tuples(my_dict: Dict[str, str]) -> list[Tuple[str, str]]:

    """Converts a dictionary into a list of two-element tuples.



    Args:

        my_dict: The dictionary to be converted.



    Returns:

        A list of two-element tuples, where each tuple consists of a key from the dictionary and the corresponding value.

    """

    return [(key, value) for key, value in my_dict.items()]

