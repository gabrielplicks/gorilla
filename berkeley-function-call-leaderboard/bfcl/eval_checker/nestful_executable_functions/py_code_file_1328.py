import urllib.parse



def url_encode(string: str) -> str:

    """Encodes a string to the required format for sending as part of a URL query string.



    Args:

        string: The string to be encoded.



    Returns:

        The encoded string.

    """

    return urllib.parse.quote(string)

