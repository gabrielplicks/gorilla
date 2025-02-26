from typing import List



def process_multiline_string(s: str) -> str:

    """Processes a multiline string by prepending word indices to each word.



    Args:

        s: The input multiline string.



    Returns:

        The processed multiline string.

    """

    lines: List[str] = s.split("\n")

    result: List[str] = []

    for i, line in enumerate(lines, 1):

        words: List[str] = line.split()

        processed_words: List[str] = [str(i) + " " + word for i, word in enumerate(words, 1)]

        result.append(" ".join(processed_words))

    return "\n".join(result)

