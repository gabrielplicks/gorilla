import re

import typing



def replace_tokens(s: str, tokens: typing.List[str], replacements: typing.List[str]):

    """

    Replaces tokens in a string with their corresponding replacements.



    Args:

        s: The string to be tokenized and replaced.

        tokens: A list of tokens to be searched in the string.

        replacements: A list of replacement strings corresponding to tokens.

    """

    for token, replacement in zip(tokens, replacements):

        s = re.sub(token, replacement, s, count=1)

    return s

