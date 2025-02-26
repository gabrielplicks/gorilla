from typing import Union



def scale_string(string: str) -> Union[float, None]:

    """Scales a string representation of a number to fit within the range [0, 1].



    Args:

        string: A string representation of a number in the range [0, 10].



    Returns:

        The scaled number as a float, or None if the input is not a valid number.

    """

    try:

        number = float(string)

        if number < 0 or number > 10:

            raise ValueError("Input number is out of range")

        scaled_number = number / 10



        return scaled_number

    except ValueError:

        return None

