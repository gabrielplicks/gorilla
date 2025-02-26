from typing import Union



def validate_alpha_underscore_string(s: Union[str, None]) -> bool:

    """Checks if a string consists only of uppercase and lowercase letters and underscores.



    Args:

        s: The input string to validate.



    Returns:

        True if the string consists only of uppercase and lowercase letters and underscores, False otherwise.

    """

    if not isinstance(s, str):

        return False



    for c in s:

        if not c.isupper() and not c.islower() and not c.isalpha() and c != "_":

            return False

    return True

