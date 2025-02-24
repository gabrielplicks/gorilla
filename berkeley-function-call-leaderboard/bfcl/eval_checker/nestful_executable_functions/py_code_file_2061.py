from typing import Dict



def count_char_frequencies(string: str) -> Dict[str, int]:

    """Counts the frequencies of each character in a string and returns this information in a dictionary.

    Args:

        string: The input string.

    Returns:

        A dictionary where each key is a character in the string and the corresponding value is the frequency of that character.

    """

    freq_dict = {char: 0 for char in set(string)}

    for char in string:

        freq_dict[char] += 1

    return freq_dict

