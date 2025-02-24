from typing import Union



def replace_plus_with_plus(input_string: Union[str, bytes]) -> Union[str, bytes]:

    """Replaces all occurrences of the character '+' with the string "plus" in the input string.



    Args:

        input_string: The input string to be modified.



    Returns:

        The modified string with all occurrences of '+' replaced with "plus".

    """

    return input_string.replace("+", "plus")

