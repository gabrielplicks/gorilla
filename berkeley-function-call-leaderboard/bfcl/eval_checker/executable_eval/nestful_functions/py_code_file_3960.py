import re



def swap_string(string: str) -> str:

    """Swaps each occurrence of "abc" in a string with "def".



    Args:

        string: The input string.



    Returns:

        A new string with each occurrence of "abc" replaced with "def".

    """

    pattern = re.compile(r'abc')

    new_string = pattern.sub(lambda m: 'def', string)

    return new_string

