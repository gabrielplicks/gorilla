import base64



def convert_base64_to_string(base64_string: str) -> str:

    """Converts a string from a valid base64 representation to the original string.



    Args:

        base64_string: The base64 string to convert.



    Returns:

        The original string.

    """

    binary_data = base64.b64decode(base64_string)

    binary_data = binary_data.rstrip(b"=")

    return binary_data.decode()

