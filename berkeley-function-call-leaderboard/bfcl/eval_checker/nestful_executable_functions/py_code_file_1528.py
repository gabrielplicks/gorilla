from typing import Union



def validate_token(token: str) -> bool:

    """Validates a token in Python by inspecting its type and following the rules for each type.



    Args:

        token: The token to validate.



    Returns:

        A Boolean value indicating whether the token is valid.

    """

    if token.startswith("123"):

        return token.isnumeric()

    elif token.startswith("\""):

        return token.count("\"") == 2

    elif token.lower() in ["true", "false"]:

        return True

    else:

        return False

