from typing import Union



def currency_number_format(number: Union[int, float], sign: str = '£') -> str:

    """Formats a number with commas and two decimal places, with an optional currency sign.



    Args:

        number: The numerical value to format.

        sign: The currency sign to include in the formatted string.



    Returns:

        The formatted number as a string.

    """

    sign = sign or ''  # If sign is None, use an empty string

    formatted_number = f'{sign}{number:,.2f}'

    return formatted_number

