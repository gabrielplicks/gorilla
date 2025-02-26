import re

import typing



def check_regex_match(regex: str, words: typing.List[str]) -> bool:

    """Checks if any of the words in the list match the given regular expression pattern.



    Args:

        regex: The regular expression pattern to match.

        words: A list of strings to check for a match.



    Returns:

        True if any of the words match the pattern, False otherwise.

    """

    for word in words:

        if re.match(regex, word):

            return True

    return False

