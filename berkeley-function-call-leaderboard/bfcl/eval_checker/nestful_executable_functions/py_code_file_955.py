from typing import Dict, Any



def rearrange_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:

    """Rearranges the keys of a nested dictionary in a specific order.



    Args:

        input_dict: The input dictionary to rearrange.



    Returns:

        A new dictionary with the same key-value pairs as the input dictionary,

        but with the keys rearranged in a specific order.

    """

    if not isinstance(input_dict, dict):

        return input_dict

    return {

        key: rearrange_dict(input_dict[key])

        for key in sorted(input_dict, reverse=True if len(input_dict) == 1 else False)

    }

