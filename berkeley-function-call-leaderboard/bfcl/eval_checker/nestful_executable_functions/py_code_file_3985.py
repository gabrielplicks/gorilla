from typing import Dict, Union



def max_value_in_nested_dictionary(dictionary: Dict[Union[int, str], Union[Dict[str, int], int]]) -> Union[int, None]:

    """Finds the maximum value in a nested dictionary.



    Args:

        dictionary: The input dictionary.



    Returns:

        The maximum value in the dictionary, or `None` if the dictionary is empty.

    """

    max_value = None

    for value in dictionary.values():

        if isinstance(value, dict):

            sub_max_value = max_value_in_nested_dictionary(value)

            if sub_max_value is not None and (max_value is None or sub_max_value > max_value):

                max_value = sub_max_value

        elif value is not None and (max_value is None or value > max_value):

            max_value = value



    return max_value

