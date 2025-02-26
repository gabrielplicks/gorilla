import re



def strip_comments_from_source_code(source_code: str) -> str:

    """Removes all comments from a source code string.



    Args:

        source_code: The source code string to remove comments from.



    Returns:

        The source code string with all comments removed.

    """

    source_code = re.sub(r'#[^\n]*', '', source_code)

    source_code = re.sub(r'"""(?:[^"\\]|\\.|"(?!""))*"""', '', source_code)

    return source_code

