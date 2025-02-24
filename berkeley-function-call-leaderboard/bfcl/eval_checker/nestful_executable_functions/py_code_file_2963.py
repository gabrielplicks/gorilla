import re



def replace_consecutive_whitespace(s: str) -> str:

    """Replaces any consecutive whitespace characters in a string with a single space.

    If the first or last character in the string is whitespace, it is removed.

    If the string consists entirely of whitespace characters, an empty string is returned.

    Args:

        s: The input string.

    """

    if not s.strip():

        return ""

    s = re.sub(r'\s+', ' ', s)

    if s[0] == ' ':

        s = s[1:]

    if s[-1] == ' ':

        s = s[:-1]

    return s

