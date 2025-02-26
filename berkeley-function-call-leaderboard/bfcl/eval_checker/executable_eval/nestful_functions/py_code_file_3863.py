import re



def contains_url(string: str) -> bool:

    """Checks if a string contains a URL.



    Args:

        string: The string to check.



    Returns:

        True if the string contains a URL, False otherwise.

    """

    return re.match(r'^https?://[^"\s]+', string) is not None

