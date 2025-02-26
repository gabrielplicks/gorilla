import urllib.parse



def encode_url(url: str) -> str:

    """Encodes a URL by replacing all non-alphanumeric and special characters with their percent-encoded representations.



    Args:

        url: The URL to be encoded.



    Returns:

        The encoded URL string.

    """

    return urllib.parse.quote_plus(url)

