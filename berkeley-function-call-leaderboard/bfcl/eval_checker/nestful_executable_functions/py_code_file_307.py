from typing import List



def tokenize_line(line: str) -> List[str]:

    """Tokenizes a line of text on whitespace and returns a list of tokens.



    Args:

        line: The line of text to tokenize.



    Returns:

        A list of tokens.

    """

    return line.split(None)

