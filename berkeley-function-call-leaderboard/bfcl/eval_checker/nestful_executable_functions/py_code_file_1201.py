from typing import BinaryIO



def find_byte_sequence(filename: str, sequence: bytes) -> int:

    """Finds the first occurrence of a byte sequence starting from the end of a file.

    Args:

        filename: The name of the file to search.

        sequence: The byte sequence to find.

    """

    with open(filename, 'rb') as f:

        f.seek(0, 2)  # Move to the end of the file

        offset = f.tell()  # Get the file size

        while offset > 0:

            f.seek(offset - 2, 0)  # Move to the previous 2 bytes

            data = f.read(2)  # Read the next 2 bytes

            if data == sequence:

                return offset - 2

            offset -= 1

    return -1  # Sequence not found

