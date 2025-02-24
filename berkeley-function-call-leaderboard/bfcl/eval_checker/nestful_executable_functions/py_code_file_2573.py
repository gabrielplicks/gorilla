from typing import List



def convert_inputs_to_ints(inputs: List[str]) -> List[int]:

    """Converts a list of user inputs into a list of integers.



    Args:

        inputs: A list of user inputs as strings.



    Raises:

        ValueError: If any of the inputs is not a string or cannot be converted to an integer.



    Returns:

        A list of integers.

    """

    result: List[int] = []

    for input in inputs:

        if not isinstance(input, str):

            raise ValueError("Input must be a string")

        try:

            result.append(int(input))

        except ValueError:

            raise ValueError("Input must be an integer")

    return result

