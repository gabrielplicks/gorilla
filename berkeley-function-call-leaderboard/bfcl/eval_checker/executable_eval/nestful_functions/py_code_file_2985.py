from typing import List



def join_tokens(tokens: List[str], separator: str) -> str:

    """Joins a list of tokens together with a given separator, where each token may contain underscores.

    If a token contains underscores, the separator is not inserted between the underscores.



    Args:

        tokens: A list of tokens to join together.

        separator: The separator to use between tokens.

    """

    if separator == "_":

        return "_".join(tokens)



    result = ""

    for token in tokens:

        if "_" in token:

            result += token + separator

        else:

            result += separator.join(token.split(separator)) + separator



    return result[:-1]

