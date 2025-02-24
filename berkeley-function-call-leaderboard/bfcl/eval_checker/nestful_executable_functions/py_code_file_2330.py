import re



def split(string: str, delimiters: List[str]) -> List[str]:

    """Splits a string into a list of strings based on multiple delimiters.



    Args:

        string: The string to split.

        delimiters: A list of delimiters (strings) to split the string on.



    Returns:

        A list of strings representing the split segments of the original string.

    """

    pattern = '|'.join(map(re.escape, delimiters))

    return re.split(pattern, string)

