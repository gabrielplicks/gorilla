from typing import Tuple



def convert_string_to_floats(s: str) -> Tuple[float, float]:

    """Converts a string in the format of "123.456 789.012" into two floating-point numbers.



    Args:

        s: The input string.



    Raises:

        ValueError: If the input string does not have exactly two parts separated by a space.

    """

    parts = s.split(' ')

    if len(parts) != 2:

        raise ValueError('Input string must have exactly two parts separated by a space')

    return (float(parts[0]), float(parts[1]))

