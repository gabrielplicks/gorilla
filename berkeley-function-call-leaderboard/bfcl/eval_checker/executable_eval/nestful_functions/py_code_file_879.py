from typing import Optional



def add_comment_header(string: str) -> Optional[str]:

    """Adds a comment header to a string.



    The comment header is a single-line comment containing the string's length.

    If the input string is empty, the function returns an empty string.



    Args:

        string: The input string.



    Returns:

        The input string with a comment header, or an empty string if the input string is empty.

    """

    stripped_string = string.strip()

    if stripped_string:

        return f'#{len(stripped_string)} {stripped_string}'

    else:

        return ''

