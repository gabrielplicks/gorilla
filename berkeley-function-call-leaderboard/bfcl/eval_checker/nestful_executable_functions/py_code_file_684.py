import base64



def convert_to_base64(string: str) -> bytes:

    """Converts a string into Base64.

    Args:

        string: The string to be converted into Base64.

    """

    return base64.b64encode(string.encode())



def decode_base64(base64_string: bytes) -> str:

    """Decodes a Base64 string back to the original string.

    Args:

        base64_string: The Base64 string to be decoded.

    """

    return base64.b64decode(base64_string).decode()

