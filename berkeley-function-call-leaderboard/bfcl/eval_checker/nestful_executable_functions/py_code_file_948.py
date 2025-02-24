from typing import Union



def validate_and_convert_numeric(value: str) -> Union[int, float, str]:

    """Validates and converts a string to its numeric representation.



    Args:

        value: The string to validate and convert.



    Returns:

        The numeric representation of the string, or an error message.

    """

    if type(value) is str:

        if value.isdigit():

            return int(value)

        elif "." in value and value.replace(".", "").isdigit():

            return float(value)

        else:

            return "Invalid numeric input"

    else:

        return "Invalid input type"

