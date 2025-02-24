import re



def find_token_prefix_regex(tokens: List[str]) -> re.Pattern:

    """Creates a regular expression that matches any string that begins with one of the input strings, case-insensitively.



    Args:

        tokens: A list of strings.



    Returns:

        A compiled regular expression pattern.

    """

    pattern = r'^(?:{})'.format('|'.join(tokens))

    return re.compile(pattern, flags=re.IGNORECASE)

