import random

random.seed(42)
import string

from typing import List



def obfuscate_string(string: str, length: int) -> str:

    """Obfuscates a string by randomly shuffling the characters and converting them to a fixed-length string representation.



    Args:

        string: The input string.

        length: The desired length of the output string.



    Returns:

        A string of the given length containing random characters from the input string.

    """

    obfuscated_string = ''

    random_characters: List[str] = random.sample(string, length)

    obfuscated_string = ''.join(random_characters)



    return obfuscated_string

