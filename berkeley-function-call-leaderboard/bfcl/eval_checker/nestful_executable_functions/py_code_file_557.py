import copy



def deep_copy_dict(dictionary: dict) -> dict:

    """Creates a deep copy of a dictionary.



    Args:

        dictionary: The dictionary to be copied.



    Returns:

        A deep copy of the input dictionary.

    """

    return copy.deepcopy(dictionary)

