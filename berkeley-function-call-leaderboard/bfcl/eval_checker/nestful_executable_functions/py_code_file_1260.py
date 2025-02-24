from typing import List, Dict



def remove_negative_numbers_and_add_to_dict(lst: List[int], key: str) -> Dict[str, int]:

    """Removes all negative numbers from a given list and adds them as values to a new dictionary.



    Args:

        lst: The list of numbers.

        key: The key to use for the dictionary.



    Returns:

        The updated dictionary.

    """

    dict_to_add = {}

    for num in lst:

        if num < 0:

            dict_to_add[key] = num

    return dict_to_add

