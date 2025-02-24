import re



def replace_backslash_sequence(s: str) -> str:

    """Replaces every backslash sequence with a single backslash in a string.



    Args:

        s: The input string.



    Returns:

        A new string with every backslash sequence replaced with a single backslash.

    """

    return re.sub(r"\\\\", r"\\", s)

