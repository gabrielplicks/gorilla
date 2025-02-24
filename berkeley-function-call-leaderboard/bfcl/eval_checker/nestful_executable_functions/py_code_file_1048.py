from typing import BinaryIO



def count_1bits_last_byte(filename: str, size: int) -> int:

    """

    Reads a binary file of a given size (in bytes) and returns the number of 1-bits in the last byte of the file.

    Args:

        filename: The name of the binary file.

        size: The size of the binary file in bytes.

    """

    with open(filename, 'rb') as f:

        f.seek(size - 1)  # Move the file pointer to the last byte

        last_byte = f.read(1)  # Read the last byte

    count = 0

    for i in range(8):

        if last_byte[0] & (1 << i):

            count += 1

    return count

