import re



def get_target_value(s: str) -> str:

    """Extracts the value between the first set of parentheses in a string.



    Args:

        s: The input string.



    Returns:

        The value between the first set of parentheses, or None if there are no parentheses.

    """

    match = re.search(r'\(([^)]+)\)', s)

    if match:

        return match.group(1)

    else:

        return None

