import re



def last_word_num(s: str) -> str:

    """Extracts the last word of a string s and returns the integer component of that word.



    Args:

        s: The input string.



    Returns:

        The integer component of the last word in the string, or "-1" if no match is found.

    """

    words = s.split()

    last_word = words[-1]

    match = re.search(r'\d+', last_word)

    if match:

        return match.group()

    return '-1'

