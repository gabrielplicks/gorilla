from io import BytesIO

from typing import Union



def convert_utf8_to_bytes(utf8_str: Union[str, bytes]) -> BytesIO:

    """

    Converts a UTF-8 encoded string to its raw bytes representation, and writes the raw bytes to a BytesIO object.

    Args:

        utf8_str: The UTF-8 encoded string to convert to raw bytes.

    """

    raw_bytes = utf8_str.encode('utf-8')

    bytes_io = BytesIO()

    bytes_io.write(raw_bytes)

    return bytes_io

