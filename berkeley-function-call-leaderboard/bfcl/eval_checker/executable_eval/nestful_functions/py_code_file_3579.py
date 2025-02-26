import urllib.parse



def url_safe_encode(string: str) -> str:

    """Creates a URL-safe version of a string by encoding it and replacing all unsafe characters with their URL-encoded equivalents.



    Args:

        string: The input string to be encoded.



    Returns:

        The URL-safe version of the input string.

    """

    return urllib.parse.quote(string)

