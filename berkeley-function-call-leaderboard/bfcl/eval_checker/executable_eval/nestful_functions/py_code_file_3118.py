import re



def replace_comment(input_string: str) -> str:

    """Removes all comment lines from a string and replaces inline comments with a single space.



    Args:

        input_string: The input string to remove comments from.



    Returns:

        The input string with all comment lines removed and inline comments replaced with a single space.

    """

    pattern = r'^(\/\/|#).*$'

    output_string = re.sub(pattern, ' ', input_string)

    return output_string

