from typing import Dict



def count_frequencies(string: str) -> Dict[str, int]:

    """Counts the frequencies of each character in a string.



    Args:

        string: The input string.



    Returns:

        A dictionary with keys being the characters and values being frequencies.

    """

    frequency_dict = {}



    for char in string:

        if char not in frequency_dict:

            frequency_dict[char] = 1

        else:

            frequency_dict[char] += 1



    return frequency_dict

