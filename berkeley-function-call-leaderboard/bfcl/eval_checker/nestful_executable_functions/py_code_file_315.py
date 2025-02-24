import base64



def encode_with_padding(string: str) -> str:

    """Encodes a string in base64 with padding to a total of 25 bytes.



    Args:

        string: The string to be encoded.

    """

    byte_array = string.encode()

    base64_string = base64.b64encode(byte_array).decode()

    padding_length = 4 - len(base64_string) % 4

    if padding_length > 0:

        base64_string += '=' * padding_length

    base64_string = base64_string.ljust(25, '=')



    return base64_string

