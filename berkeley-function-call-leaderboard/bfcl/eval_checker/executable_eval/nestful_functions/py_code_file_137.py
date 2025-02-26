from typing import Union



def is_decimal_string(s: Union[str, int, float]) -> bool:

    """Checks if the input string can be converted to a non-negative decimal integer.



    Args:

        s: The input string to be checked.



    Returns:

        True if the input string can be converted to a non-negative decimal integer, False otherwise.

    """

    if not isinstance(s, str):

        return False

    if not s.isdigit() or s == "":

        return False

    if s.strip() != s:

        return False

    if s[0] == '0' and len(s) > 1:

        return False

    if s[0] == '-':

        return False

    try:

        int(s)

        return True

    except ValueError:

        return False

