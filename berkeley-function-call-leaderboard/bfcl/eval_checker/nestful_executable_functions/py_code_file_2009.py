import base64



def convert_to_base64_url(byte_array: bytes) -> str:

    """

    Converts a byte array to a Base64-encoded string using the URL-safe alphabet.



    Args:

        byte_array: A byte array to be converted to Base64.

    """

    base64_str = base64.b64encode(byte_array).decode('utf-8').replace('=', '')

    return base64_str

