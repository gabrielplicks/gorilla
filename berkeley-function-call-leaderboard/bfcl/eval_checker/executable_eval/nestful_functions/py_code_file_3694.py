import base64



def urlsafe_base64_encode(text: str) -> str:

    """Generates a base64 encoding of a given string, where the generated encoding is URL-safe and uses a `-` and `_` character set instead of the default `+` and `/` characters.



    Args:

        text: The input string to be encoded.



    Returns:

        A URL-safe base64 encoded string.

    """

    byte_string = text.encode()

    encoded_string = base64.b64encode(byte_string, altchars=b'-_').decode()



    return encoded_string

