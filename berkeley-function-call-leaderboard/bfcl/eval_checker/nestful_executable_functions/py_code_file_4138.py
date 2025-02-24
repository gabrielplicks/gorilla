import re



def replace_a_with_e(string: str) -> str:

    """

    Replaces all occurrences of 'a' with 'e' in the given string.

    Args:

        string: The input string.

    """

    return re.sub(r'a', 'e', string)

