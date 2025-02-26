from typing import Union



def shift_letters(input_string: str) -> str:

    """

    Replaces every letter in the input string with the letter following it in the alphabet.

    The function does not change the case of the letter, and loops back to 'a' from 'z'.



    Args:

        input_string: The input string.



    Returns:

        The string with every letter replaced by the letter following it in the alphabet.

    """

    output_string = ""

    for letter in input_string:

        letter_value = ord(letter)

        new_value = letter_value + 1

        new_letter = chr(new_value)

        output_string += new_letter

    return output_string

