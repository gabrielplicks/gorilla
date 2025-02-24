import base64



def byte_string_to_base64(byte_string: bytes) -> str:

    """Encodes a byte string to its equivalent base64 encoded string.



    Args:

        byte_string: The byte string to be encoded.

    """

    encoded_bytes = base64.b64encode(byte_string)

    return encoded_bytes.decode('ascii')

