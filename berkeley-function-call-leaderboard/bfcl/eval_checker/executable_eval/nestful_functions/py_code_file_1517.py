from typing import Union



def convert_number_to_string(number: Union[int, float]) -> str:

    """Converts a number to a string and adds the appropriate suffix.



    Args:

        number: The number to convert.



    Returns:

        The modified string with the appropriate suffix.

    """

    suffixes = {10**3: "thousand", 10**6: "million", 10**9: "billion", 10**12: "trillion"}

    if number in suffixes:

        return f"{number} {suffixes[number]}"

    else:

        return str(number)

