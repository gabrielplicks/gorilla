from typing import BinaryIO



def recognize_magic_number(path: str) -> str:

    """Recognizes the magic number of a binary file.



    Args:

        path: The path to the binary file.



    Returns:

        The name of the file type if the binary file starts with a magic number,

        otherwise an empty string.

    """

    with open(path, "rb") as f:

        magic_number = f.read(4)



    if magic_number == b"\x1F\x8B":

        return "gzipped file"

    elif magic_number in (b"\xFF\xD8\xFF\xE0", b"\x47\x49\x46\x38", b"\x89\x50\x4E\x47"):

        return "image file"

    else:

        return "regular file"

