from typing import Optional



def letters_in_order(string: str) -> bool:

    """Detects if a string contains the letters 'a', 'b', and 'c' in that order.

    Args:

        string: The input string.

    Returns:

        A boolean value indicating whether the string contains the letters 'a', 'b', and 'c' in that order.

    """

    last_letter: Optional[str] = ''

    for letter in string:

        if letter == 'a':

            last_letter = letter

        elif letter == 'b' and last_letter == 'a':

            last_letter = letter

        elif letter == 'c' and last_letter == 'b':

            return True

        else:

            return False

    return False

