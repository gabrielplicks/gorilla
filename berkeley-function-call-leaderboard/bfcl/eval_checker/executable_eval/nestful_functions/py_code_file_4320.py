from typing import Dict



def replace_with_escapes(string: str, escape_dict: Dict[str, str]) -> str:

    """Replaces all occurrences of the characters in the passed dictionary with the corresponding escaped values.

    Args:

        string: The input string.

        escape_dict: A dictionary of the form {'char': 'escape_val'}, where 'char' is a character to be replaced and 'escape_val' is its corresponding escaped value.

    """

    replaced_string = ''

    for char in string:

        if char in escape_dict:

            replaced_string += escape_dict[char]

        else:

            replaced_string += char

    return replaced_string

