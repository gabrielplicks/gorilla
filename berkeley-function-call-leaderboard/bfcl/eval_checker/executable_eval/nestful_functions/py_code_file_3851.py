from typing import Union



def is_comment_line(line: Union[str, bytes]) -> bool:

    """Determines if a given line is a comment line in a programming language.



    Args:

        line: The line to check.



    Returns:

        True if the line is a comment line, False otherwise.

    """

    line = line.strip()

    comment_prefixes = ['#', '//', '/*', '<!--']

    for prefix in comment_prefixes:

        if line.startswith(prefix):

            return True

    if not line:

        return True

    return False

