from typing import Union



def add_to_end_of_string(string: str, char: Union[str, int]) -> str:

    """Adds a character to the end of each character in a string.



    Args:

        string: The input string.

        char: The character to be added to the end of each character in the string.

    """

    new_string = ""



    for character in string:

        new_string += character + str(char)



    return new_string

