import random

random.seed(42)
from typing import List



def random_case(text: str) -> str:

    """Randomizes the case of alphabetic characters in a string.



    Args:

        text: The input string.



    Returns:

        A new string with random uppercase and lowercase letters.

    """

    case_list = ['upper', 'lower']

    modified_letters: List[str] = []

    for char in text:

        if not char.isalpha():

            modified_letters.append(char)

            continue

        modified_letter = random.choice(case_list)

        if modified_letter == 'upper':

            modified_letters.append(char.upper())

        else:

            modified_letters.append(char.lower())

    return ''.join(modified_letters)

