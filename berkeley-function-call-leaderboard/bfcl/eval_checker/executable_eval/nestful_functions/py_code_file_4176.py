from typing import Any



def access_byte(buffer: bytes, index: int) -> int:

    """Accesses a specific byte within a memory buffer using advanced pointer arithmetic.

    Args:

        buffer: The buffer containing raw bytes.

        index: A 16-bit integer specifying the index of the desired byte within the buffer.

    """

    byte_index = index * 2

    byte_value = buffer[byte_index]

    return byte_value

