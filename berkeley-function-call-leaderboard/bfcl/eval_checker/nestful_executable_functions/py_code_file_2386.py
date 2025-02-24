def str_to_num(s: str) -> int:

    """

    Converts a string to a number. If the string is a decimal number, it returns

    the corresponding integer value. If the string is a hexadecimal number, it

    returns the corresponding integer value converted from the hexadecimal.

    Args:

        s: The input string to be converted to a number.

    """

    return int(s, base=16 if s.startswith("0x") else 10)

