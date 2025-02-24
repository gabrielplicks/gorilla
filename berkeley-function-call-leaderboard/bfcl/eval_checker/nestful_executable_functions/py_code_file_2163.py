import base64

from typing import Union



def image_to_base64(path_or_bytes: Union[str, bytes]) -> str:

    """Converts an image file to a base64-encoded string.



    Args:

        path_or_bytes: The file path or BytesIO object of the image.



    Returns:

        The base64-encoded string of the image.

    """

    if isinstance(path_or_bytes, str):

        with open(path_or_bytes, 'rb') as f:

            data = f.read()

    else:

        data = path_or_bytes



    return base64.b64encode(data).decode('utf-8')

