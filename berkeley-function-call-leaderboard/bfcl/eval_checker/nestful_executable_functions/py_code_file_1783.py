from typing import Tuple



def count_digits_alphabets_special_characters(text: str) -> Tuple[int, int, int]:

    """

    Counts the number of digits, alphabets, and special characters in a given string.



    Args:

        text: The input string.



    Returns:

        A tuple containing the total number of digits, alphabets, and special characters in the string.

    """

    num_digits = 0

    num_alphabets = 0

    num_special_characters = 0



    for char in text:

        if char.isdigit():

            num_digits += 1

        elif char.isalpha():

            num_alphabets += 1

        elif not char.isalnum():

            num_special_characters += 1



    return num_digits, num_alphabets, num_special_characters

