from typing import Dict



def check_dict1(input_dict: Dict[str, int]) -> bool:

    """Checks if the dictionary contains both the key "a" and value 1, or if the dictionary contains both the key "b" and value 2.



    Args:

        input_dict: The input dictionary.



    Returns:

        True if the dictionary contains either of the key-value pairs, False otherwise.

    """

    return (input_dict.get("a") == 1 and input_dict.get("b") == 2) or (input_dict.get("a") == 2 and input_dict.get("b") == 1)

