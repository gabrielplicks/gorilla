from typing import Dict



def sort_and_print_dictionary(dictionary: Dict[str, int]) -> str:

    """Sorts the keys of a dictionary and returns a string with the key-value pairs.



    Args:

        dictionary: The dictionary to sort and print.



    Returns:

        A string with the key-value pairs in sorted order.

    """

    sorted_keys = sorted(dictionary.keys())

    output_string = ''

    for key in sorted_keys:

        value = dictionary[key]

        output_string += f'{key}={value}, '

    return output_string[:-2]

