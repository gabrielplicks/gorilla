from typing import Union



def convert_int_to_string_with_base(num: int, base: Union[str, int]) -> str:

    """Converts an integer into a string with a given base.

    Args:

        num: The integer to be converted.

        base: The base to convert the integer to.

    """

    return f"{num:b}" if base == "binary" else f"{num:o}" if base == "octal" else f"{num:d}" if base == "decimal" else f"{num:x}"

