from typing import List



def convert_list_format(strings: List[str]) -> List[str]:

    """Converts a list of strings into a new list with a specific format.

    Given a list of strings, where each string has the format `("Name", "value")`,

    convert each string to a new string with the format `"Name: value"`,

    where the string `"value"` is surrounded by double quotes.

    Args:

        strings: A list of strings to be converted.

    """

    return [f"{name}: '{value}'" for name, value in strings]

