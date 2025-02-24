from typing import List



def reverse_line_by_word(line: str) -> str:

    """Reverses the words in a line without reversing the entire line.



    Args:

        line: The input line.



    Returns:

        The line with the words reversed.

    """

    words: List[str] = line.split()

    reversed_words: List[str] = [word[::-1] for word in words]

    return " ".join(reversed_words)

