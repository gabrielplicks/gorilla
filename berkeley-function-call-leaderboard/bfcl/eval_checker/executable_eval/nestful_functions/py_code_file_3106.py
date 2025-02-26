from typing import Union



def string_to_boolean(s: str) -> Union[bool, None]:

    """Converts a string to a boolean value.



    Args:

        s: The input string.



    Returns:

        True if the string is "True" (case-insensitive), False if the string is "False" (case-insensitive),

        None if the string is neither.

    """

    if s.lower() == 'true':

        return True

    elif s.lower() == 'false':

        return False

    else:

        return None if s.isdigit() else None

