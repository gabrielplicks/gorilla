from typing import AnyStr



def insert_comma_space(input_string: AnyStr) -> AnyStr:

    """Inserts a comma and space between each character in the input string.

    Args:

        input_string: The input string.

    """

    output_string = ""

    for char in input_string:

        output_string += char + ", "

    return output_string[:-2]  # Remove the last comma and space

