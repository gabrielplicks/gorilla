from typing import List



def remove_non_letters(input_string: str) -> List[str]:

    """

    Removes all characters that are not letters or digits from a string and returns a list of strings.

    Args:

        input_string: The input string to remove non-letter characters from.

    """

    output_string = ""

    for char in input_string:

        if char.isalpha() or char.isdigit():

            output_string += char

    return [output_string]

