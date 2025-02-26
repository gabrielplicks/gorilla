import re



def split_on_camel_case(s: str) -> list[str]:

    """Splits a string into a list of words based on camel case.



    Args:

        s: The input string.



    Returns:

        A list of words.

    """

    return re.findall(r"[A-Z]?[a-z]+|[A-Z]+", s)

