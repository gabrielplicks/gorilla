from typing import Tuple



def convert_to_roman_numeral(number: int) -> str:

    """Converts an integer to a Roman numeral string.



    Args:

        number: The integer to be converted.



    Returns:

        The Roman numeral string representation of the input integer.

    """

    symbols: Tuple[Tuple[str, int], ...] = (('M', 1000), ('CM', 900), ('D', 500), ('CD', 400), ('C', 100), ('XC', 90), ('L', 50), ('XL', 40), ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1))

    result: str = ''

    current: int = number

    for symbol, value in symbols:

        while current >= value:

            result += symbol

            current -= value

    return result

