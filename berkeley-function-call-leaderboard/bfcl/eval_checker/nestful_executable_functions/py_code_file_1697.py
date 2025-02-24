from typing import Any



def match_key_value(string: str) -> str:

    """

    Matches each key-value pair in the input string and concatenates them in the format: `A=B&C=D&E=F&G=H`.



    Args:

        string: The input string, which is a concatenation of key-value pairs separated by a comma (`,`).



    Returns:

        The output string, which is a concatenation of key-value pairs in the desired format.

    """

    key_value_pairs = string.split(',')

    output = ''

    for key_value_pair in key_value_pairs:

        key, value = key_value_pair.split('=')

        output += f'{key}={value}&'

    output = output[:-1]



    return output

