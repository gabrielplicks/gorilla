import re



def replace_if_match(pattern: str, old: str, new: str) -> str:

    """Replaces all occurrences of `old` that match the given pattern with `new`.



    Args:

        pattern: The regular expression pattern to match.

        old: The string to search for matches.

        new: The string to replace the matches with.

    """

    if re.search(pattern, old):

        return re.sub(pattern, new, old)

    return old

