import base64



def base64_encode_str(byte_str: bytes) -> str:

    """Encodes a byte string to a Base64-encoded string.



    Args:

        byte_str: The byte string to encode.



    Returns:

        The Base64-encoded string.

    """

    return base64.b64encode(byte_str).decode('utf-8')

