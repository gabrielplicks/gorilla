import json

from typing import Any



def tokens_equal(token1: str, token2: str) -> bool:

    """Determines if two tokens are equal in structure.



    Args:

        token1: The first token.

        token2: The second token.



    Returns:

        True if the tokens are equal in structure, False otherwise.

    """

    json1 = json.loads(token1)

    json2 = json.loads(token2)

    if json1.keys() != json2.keys():

        return False

    for key in json1.keys():

        if json1[key] != json2[key]:

            return False

    return True

