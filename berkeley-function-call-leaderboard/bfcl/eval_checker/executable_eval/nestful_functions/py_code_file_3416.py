import re



def count_cats(text: str) -> int:

    """Counts the number of occurrences of the substring "cat" in a string.



    The function uses regular expressions to make the matching case-insensitive.

    It also handles empty or non-string inputs, returning 0 in those cases.



    Args:

        text: The input string.



    Returns:

        The number of occurrences of "cat" in the input string.

    """

    if not isinstance(text, str):

        return 0



    pattern = re.compile(r"(?i)cat")

    matches = re.findall(pattern, text)

    return len(matches)

