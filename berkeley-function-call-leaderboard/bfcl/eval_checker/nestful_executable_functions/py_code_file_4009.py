import re

from typing import Dict



def decrypt_token_string(token_string: str, variables: Dict[str, str]) -> str:

    """Decrypts a token string containing variable names.



    Args:

        token_string: The token string to decrypt.

        variables: A dictionary of variable names and their corresponding values.



    Returns:

        The decrypted string.

    """

    pattern = re.compile(r'{(\w+)}')

    decrypted_string = pattern.sub(lambda match: variables.get(match.group(1), match.group()), token_string)



    return decrypted_string

