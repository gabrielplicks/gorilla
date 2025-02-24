import re  # import the regular expression module



def matches_pattern(string: str, pattern: str) -> bool:

    """Checks if a string matches a given regex pattern.



    Args:

        string: The string to be matched.

        pattern: The regex pattern to match against.

    """

    match = re.match(pattern, string)

    return bool(match)

