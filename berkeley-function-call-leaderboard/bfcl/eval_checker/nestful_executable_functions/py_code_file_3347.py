from typing import Union



def extract_decimal_from_string(input_string: str) -> Union[float, None]:

    """Extracts the decimal value from a string.



    Args:

        input_string: The input string.



    Returns:

        The decimal value as a float, or None if the input string does not contain a decimal value.



    Raises:

        TypeError: If the input is not a string.

        ValueError: If the input string does not contain a decimal value.

    """

    if not isinstance(input_string, str):

        raise TypeError(f"{input_string} is not a string.")

    parts = input_string.split(".")

    if len(parts) < 2:

        raise ValueError(f"{input_string} does not contain a decimal value.")

    return float(input_string)

