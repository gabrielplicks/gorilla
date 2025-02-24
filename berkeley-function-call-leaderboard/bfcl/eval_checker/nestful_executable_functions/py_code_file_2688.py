import re



def split_by_camel_case(string: str) -> list:

    """Splits a string into a list of words based on camel case.



    Args:

        string: The string to be split.



    Returns:

        A list of words in the string.

    """

    words = re.split(r'(?=[A-Z])', string)

    return [word.lower() for word in words]

