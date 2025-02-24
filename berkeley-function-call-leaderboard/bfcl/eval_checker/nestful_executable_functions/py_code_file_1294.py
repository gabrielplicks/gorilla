from typing import Dict



def dict_values_to_lists(input_dict: Dict[str, int]) -> Dict[str, list]:

    """Transforms a dictionary of values into a dictionary of lists.

    The keys of the input dictionary are the same as the keys of the output dictionary,

    and the values are lists of the items at each key from the input dictionary.

    Args:

        input_dict: The input dictionary.

    Returns:

        A dictionary of lists.

    """

    return {k: [v] for k, v in input_dict.items()}

