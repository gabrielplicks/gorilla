from typing import Dict, List



def transform(string_list: List[str]) -> Dict[str, str]:

    """Transforms a list of strings into a dictionary.



    Each string in the list is expected to be in the format "key=value".

    The function splits each string by the "=" character and uses the

    left side as the key and the right side as the value in the resulting

    dictionary.



    Args:

        string_list: A list of strings in the format "key=value".



    Returns:

        A dictionary where the keys and values are derived from the input

        string list.

    """

    result = {}

    for string in string_list:

        key, value = string.split('=')

        result[key] = value

    return result

