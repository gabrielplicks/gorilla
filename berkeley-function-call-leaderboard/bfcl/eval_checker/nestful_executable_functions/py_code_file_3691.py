from typing import Union



def count_upper(string: Union[str, int]) -> Union[int, str]:

    """Counts the number of uppercase letters in a string.

    Args:

        string: The input string.

    Returns:

        The number of uppercase letters in the string, or an error message if the input is not a string.

    """

    try:

        count = 0

        for char in string:

            if char.isupper():

                count += 1

        return count

    except TypeError:

        return "Input must be a string."

