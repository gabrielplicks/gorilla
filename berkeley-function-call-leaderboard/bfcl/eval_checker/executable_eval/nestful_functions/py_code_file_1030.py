from typing import Dict



def simplify_dict(nested_dict: Dict, prefix: str = "") -> Dict:

    """Simplifies a nested dictionary into a dictionary with only the leaf values.

    The simplified dictionary's keys are the path to each leaf's value in the nested dictionary.



    Args:

        nested_dict: The nested dictionary to simplify.

        prefix: The prefix to use for the keys in the simplified dictionary.



    Returns:

        The simplified dictionary.

    """

    simplified_dict = {}

    for key, value in nested_dict.items():

        if isinstance(value, dict):

            simplified_dict.update(simplify_dict(value, prefix + key + "."))

        else:

            simplified_dict[prefix + key] = value

    return simplified_dict

