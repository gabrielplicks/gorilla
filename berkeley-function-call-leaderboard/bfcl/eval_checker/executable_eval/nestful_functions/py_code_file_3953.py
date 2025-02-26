from typing import List, Union



def convert_strings_to_integers(strings: List[str]) -> List[Union[int, None]]:

    """Converts a list of strings to a list of integers.

    If the conversion is not possible, the function returns None.

    Args:

        strings: A list of strings to be converted to integers.

    Returns:

        A list of integers or None.

    """

    integers = []

    for string in strings:

        try:

            converted_integer = int(string, base=10)

        except ValueError:

            converted_integer = None

        integers.append(converted_integer)

    return integers

