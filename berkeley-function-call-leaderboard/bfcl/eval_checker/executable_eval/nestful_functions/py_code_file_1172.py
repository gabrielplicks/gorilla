from typing import Union



def print_string(string: str, upper: bool = False) -> None:

    """Prints a string in either upper or lower case.



    Args:

        string: The string to display.

        upper: A boolean indicating whether the string should be printed in upper case.

            Defaults to False.

    """

    if upper:

        print(string.upper())

    else:

        print(string.lower())

