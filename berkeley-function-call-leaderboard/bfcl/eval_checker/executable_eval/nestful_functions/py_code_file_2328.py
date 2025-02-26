from typing import Union



def strip_and_replace_spaces(s: Union[str, bytes]) -> str:

    """

    Removes the leading and trailing white spaces from a given string.

    If the string contains multiple consecutive white spaces, then replace them with a single space.

    Raises an exception if the string contains non-ASCII characters.



    Args:

        s: The input string.

    """

    try:

        return s.strip().replace('  ', ' ')  # Replace consecutive white spaces with a single space

    except UnicodeEncodeError:

        raise Exception("String contains non-ASCII characters.")

