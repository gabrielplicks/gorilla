from typing import BinaryIO



def read_image_file(path: str) -> bytes:

    """Reads an image file from the given path and returns its contents as a byte string.

    Args:

        path: The path to the image file.

    """

    with open(path, 'rb') as f:

        contents = f.read()

    return contents

