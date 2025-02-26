from typing import Union



def encode_bool(value: Union[bool, int]) -> int:

    """Encodes a boolean value into an integer of length 16.

    Args:

        value: The boolean value to be encoded.

    """

    encoded_value = 0b1111111111111111 if value else 0b0000000000000000

    return encoded_value



def decode_bool(encoded_value: int) -> int:

    """Decodes a binary string back into its original integer.

    Args:

        encoded_value: The encoded integer value.

    """

    decoded_value = 1 if encoded_value == 0b1111111111111111 else 0

    return decoded_value

