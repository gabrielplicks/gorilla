from typing import Dict



def all_letters_appear_twice(s: str) -> bool:

    """Checks if all letters in the input string appear exactly twice.



    Args:

        s: The input string.



    Returns:

        True if all letters appear exactly twice, False otherwise.

    """

    letter_freq: Dict[str, int] = {}

    for letter in s:

        letter_freq[letter] = letter_freq.get(letter, 0) + 1

    return all(freq == 2 for freq in letter_freq.values())

