from typing import List



def detokenize(token_list: List[str]) -> str:

    """Detokenizes a list of tokens into a string.



    Args:

        token_list: A list of tokens.



    Returns:

        A string.

    """

    tokens = []

    for token in token_list:

        if token == "[SEP]":

            tokens.append(".")

        elif token == "[MASK]":

            tokens.append("[MASK]")

        else:

            tokens.append(token)

    return " ".join(tokens)

