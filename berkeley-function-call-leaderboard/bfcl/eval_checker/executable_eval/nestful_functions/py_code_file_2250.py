from typing import Dict



def extract_frequencies_of_chars(s: str) -> Dict[str, int]:

    """Extracts the frequencies of characters in a given string.



    Args:

        s: The input string.



    Returns:

        A dictionary that maps each character in `s` to the number of times that character appears in `s`.

    """

    frequencies = {}

    for char in s:

        if char not in frequencies:

            frequencies[char] = 0

        frequencies[char] += 1

    return frequencies

