import base64

from typing import List, Union



def byte_to_base64(byte_string: Union[bytes, bytearray]) -> List[bytes]:

    """Converts a byte string into a list of bytes, where each byte is its corresponding base64 encoding.

    Args:

        byte_string: The byte string to be converted.

    """

    return [base64.b64encode(byte) for byte in byte_string]

