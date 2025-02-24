import re



def replace_seven(input_string: str) -> str:

    """Replaces all occurrences of 7 with 1 in a given string.



    Args:

        input_string: The input string.



    Returns:

        The input string with all occurrences of 7 replaced with 1.

    """

    return re.sub(r'7', '1', input_string)

