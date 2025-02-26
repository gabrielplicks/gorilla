from typing import Dict



def most_common_letter(s: str) -> str:

    """

    Returns the most common letter in the input string.

    If there is a tie, returns the letter that comes first alphabetically.



    Args:

        s: The input string.



    Returns:

        The most common letter in the input string.

    """

    letter_counts: Dict[str, int] = {}

    for letter in s:

        if letter not in letter_counts:

            letter_counts[letter] = 0

        letter_counts[letter] += 1



    most_common_letter = max(letter_counts, key=letter_counts.get)

    return most_common_letter

