from typing import Union



def replace_dogs_with_cats(input_string: str) -> str:

    """Replaces all instances of the word 'dog' with 'cat' in the input string.

    If the word is followed by 's' (e.g. 'dogs'), it is replaced by 'cats'.

    Args:

        input_string: The input string.

    Returns:

        A new string with all instances of 'dog' replaced by 'cat'.

    """

    if input_string.startswith("dogs"):

        new_string = input_string.replace("dogs", "cats")

    else:

        new_string = input_string.replace("dog", "cat")

    return new_string

