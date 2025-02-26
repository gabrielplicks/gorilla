from typing import Dict



def remove_elements_less_than(dictionary: Dict[str, int], threshold: int) -> None:

    """Removes all elements from a dictionary, whose values are numbers, that are less than a given threshold.

    The function also removes the corresponding key-value pair from the original dictionary.

    Args:

        dictionary: The dictionary to remove elements from.

        threshold: The threshold value.

    """

    keys_to_remove = []

    for key, value in dictionary.items():

        if value < threshold:

            keys_to_remove.append(key)

    for key in keys_to_remove:

        del dictionary[key]

