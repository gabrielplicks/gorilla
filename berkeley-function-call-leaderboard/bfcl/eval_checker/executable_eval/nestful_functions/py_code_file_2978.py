from typing import List



def max_string_value(words: List[str], values: List[int]) -> int:

    """Calculates the maximum possible value of any string in the list of words.



    Each letter's value is the index (1-indexed) of its position in the alphabet.

    For example, a = 1, b = 2, c = 3, and so on.



    Args:

        words: A list of strings containing only lowercase alphabets.

        values: A list of integers representing the corresponding values of the letters.



    Returns:

        The maximum possible value of any string in the list of words.

    """

    max_value = 0

    for word in words:

        value = 0

        for letter in word:

            value += values[ord(letter) - ord('a')]

        max_value = max(max_value, value)

    return max_value

