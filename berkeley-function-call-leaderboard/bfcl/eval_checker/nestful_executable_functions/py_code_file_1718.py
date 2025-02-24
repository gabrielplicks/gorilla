from typing import List, Dict



def create_dictionary_from_string_list(input_list: List[str]) -> Dict[str, str]:

    """Creates a dictionary from a list of string-type keys and values (key-value pairs).

    The input strings are in the format 'key=value'.

    Args:

        input_list: A list of strings in the format 'key=value'.

    Returns:

        A dictionary where the keys are the keys from the input strings and the values are the values from the input strings.

    """

    output_dictionary = {}



    for string in input_list:

        key_value = string.split('=')

        output_dictionary[key_value[0]] = key_value[1]



    return output_dictionary

