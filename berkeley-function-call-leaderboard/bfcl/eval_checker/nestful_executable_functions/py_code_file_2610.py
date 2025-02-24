from typing import List



def tokenize_and_convert(input_string: str) -> str:

    """Tokenizes a string input and converts each word to an integer or string based on its type.



    Args:

        input_string: The input string to tokenize and convert.



    Returns:

        The input string with each word converted to an integer or string based on its type.

    """

    tokens: List[Union[int, str]] = []

    for word in input_string.split():

        try:

            tokens.append(int(word))

        except ValueError:

            tokens.append(word)

    return " ".join(map(str, tokens))

