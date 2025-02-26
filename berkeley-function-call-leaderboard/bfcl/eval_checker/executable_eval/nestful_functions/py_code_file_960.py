from typing import Union



def parse_number_or_string(s: str) -> Union[int, float, str, str]:

    """Parses a string and returns its corresponding number if it is a valid integer or float, its value unchanged if it is a valid string, or an error message if it is not a valid number or string.



    Args:

        s: The input string.



    Returns:

        The corresponding number if the input is a valid integer or float, the value unchanged if it is a valid string, or an error message if it is not a valid number or string.

    """

    try:

        return int(s)

    except ValueError:

        try:

            return float(s)

        except ValueError:

            if s.isalpha():

                return s

            else:

                return "Invalid input"

