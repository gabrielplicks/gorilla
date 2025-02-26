import re



def input_to_regex(input_string: str) -> str:

    """Creates a regular expression that matches the given input string.

    Args:

        input_string: The input string to be converted into a regular expression.

    """

    regex = re.escape(input_string)

    return r'^{}$'.format(regex)

