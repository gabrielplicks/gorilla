import re

import typing



def join_all_tokens(string: str) -> str:

    """Concatenates all tokens from the input string and returns a new string with each token separated by a space.

    Args:

        string: The input string.

    """

    tokens: typing.List[str] = re.findall(r"\w+|[^\w\s]", string)

    return " ".join(tokens)

