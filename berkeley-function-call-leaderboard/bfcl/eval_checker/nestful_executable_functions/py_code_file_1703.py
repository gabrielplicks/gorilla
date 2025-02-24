import re



def match_regex_any_type(pattern: str, target: Union[str, bytes]) -> bool:

    """

    Checks if a target string matches a given regular expression pattern.

    Handles both string and bytes types as input.



    Args:

        pattern: The regular expression pattern to match.

        target: The target string to match against the pattern.

    """

    if isinstance(target, str):

        regex = re.compile(pattern)

        return regex.fullmatch(target) is not None

    elif isinstance(target, bytes):

        target_str = target.decode("utf-8")

        regex = re.compile(pattern, re.ASCII)

        return regex.fullmatch(target_str) is not None

    else:

        return False

