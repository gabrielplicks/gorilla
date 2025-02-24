from typing import Dict



def remove_numeric_keys(input_dict: Dict[str, object]) -> Dict[str, object]:

    """Removes keys with values that are numeric types from a dictionary.



    Args:

        input_dict: The input dictionary.



    Returns:

        A new dictionary with only the keys that have non-numeric values.

    """

    return {k: v for k, v in input_dict.items() if not isinstance(v, (int, float, complex))}

