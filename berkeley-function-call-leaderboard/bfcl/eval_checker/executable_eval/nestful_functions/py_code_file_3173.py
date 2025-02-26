import ast



def parse_tuple(string: str) -> tuple:

    """Parses a string as a Python tuple.



    Args:

        string: The string to parse.



    Returns:

        The parsed tuple, or `None` if the input string is invalid or doesn't represent a tuple.

    """

    if not string.startswith("(") or not string.endswith(")") or string.count("(") != string.count(")"):

        return None

    try:

        return ast.literal_eval(string)

    except ValueError:

        return None

