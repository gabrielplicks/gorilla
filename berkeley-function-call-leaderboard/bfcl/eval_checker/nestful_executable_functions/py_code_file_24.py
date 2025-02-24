from typing import Union



def parse_libsvm_label(input_string: str) -> int:

    """Parses the first character of a libsvm file and converts it to a numeric label.



    Args:

        input_string: The input string to parse.



    Returns:

        The label as an integer.



    Raises:

        ValueError: If the input is not a string or if the first character is not a digit.

    """

    try:

        if not isinstance(input_string, str):

            raise ValueError("Input must be a string")

        label_string = input_string.split(" ")[0]

        if not label_string[0].isdigit():

            raise ValueError("Invalid input")

        label = int(label_string[0])

        return label

    except Exception as e:

        raise e

