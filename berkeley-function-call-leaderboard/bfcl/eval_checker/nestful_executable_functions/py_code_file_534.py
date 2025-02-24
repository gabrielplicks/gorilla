import math



def validate_and_sqrt(input_str: str) -> float:

    """

    Prompts the user to enter a number and returns the square root of the number.

    The function validates the input and ensures the user enters a non-negative number.

    If the input is invalid, the function raises a ValueError with a custom message.



    Args:

        input_str: The input string to be converted to a number.

    """

    try:

        number = float(input_str)

        if math.isfinite(number) and number >= 0:

            return math.sqrt(number)

        else:

            raise ValueError('Invalid input. Please enter a non-negative number.')

    except ValueError:

        raise ValueError('Invalid input. Please enter a number.')

