import re



def is_alphanumeric_or_hyphen(string: str) -> bool:

    """

    Checks whether a string is alphanumeric or hyphen.



    Args:

        string: The string to check.



    Returns:

        True if the string contains only alphanumeric characters (letters and numbers) or hyphens (-), False otherwise.

    """

    pattern = re.compile(r"^[a-zA-Z0-9-]+$")

    return pattern.match(string) is not None

