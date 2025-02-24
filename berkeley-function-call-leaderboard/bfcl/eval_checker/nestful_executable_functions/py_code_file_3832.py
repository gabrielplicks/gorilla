from typing import Dict



def find_char_frequency(string: str) -> Dict[str, int]:

    """Calculates the frequency of each character in a string, excluding spaces.



    Args:

        string: The input string.



    Returns:

        A dictionary containing the frequency of each character in the string.

    """

    char_freq = {}

    for char in string:

        if char != ' ':

            if char in char_freq:

                char_freq[char] += 1

            else:

                char_freq[char] = 1

    return char_freq

