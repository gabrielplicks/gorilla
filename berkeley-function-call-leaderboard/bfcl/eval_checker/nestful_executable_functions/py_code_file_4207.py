import re



def add_commas_to_string(s: str) -> str:

    """Adds commas to the beginning and end of a string, and after each word.

    Leaves commas in the string if they appear in the word itself.

    Args:

        s: The input string.

    """

    s = re.sub(r",\s*", " , ", s)

    return f",{s},"

