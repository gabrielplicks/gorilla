import re



def count_occurrences_lazy(pattern: str, string: str) -> int:

    """

    Counts the number of occurrences of a regular expression pattern in a string in lazy mode.

    Args:

        pattern: The regular expression pattern to match.

        string: The string to search in.

    """

    return len(re.findall(pattern, string))



def count_occurrences_greedy(pattern: str, string: str) -> int:

    """

    Counts the number of occurrences of a regular expression pattern in a string in greedy mode.

    Args:

        pattern: The regular expression pattern to match.

        string: The string to search in.

    """

    pattern = re.compile(pattern)

    return len(pattern.findall(string))

