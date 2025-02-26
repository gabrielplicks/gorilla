import re

import typing



def regexp_match(regexp: str, string: str) -> typing.List[str]:

    """Matches the regular expression against the string and returns a list of matches.

    If there are no matches, the function returns an empty list.

    Args:

        regexp: The regular expression to match.

        string: The string to match against.

    """

    return re.findall(regexp, string)

