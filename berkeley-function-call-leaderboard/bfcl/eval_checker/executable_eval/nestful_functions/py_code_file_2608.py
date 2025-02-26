from typing import List



def reverse_hex_bytes(hex_str: str) -> str:

    """Reverses the byte ordering of a hexadecimal string.

    Args:

        hex_str: The hexadecimal string to reverse.

    """

    byte_pairs: List[str] = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]

    reversed_pairs: List[str] = byte_pairs[::-1]

    reversed_hex_str: str = ''.join(reversed_pairs)

    return reversed_hex_str

