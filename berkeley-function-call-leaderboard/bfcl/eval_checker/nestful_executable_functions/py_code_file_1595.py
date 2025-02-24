from typing import Dict



def bitmask_to_dict(mask_string: str) -> Dict[int, bool]:

    """

    Converts a string representing a set of bit masks into a dictionary.

    The string consists of a comma-separated list of bit indices, where each index represents a bit position in a bit mask.

    The dictionary has each bit index as a key and a boolean value for its bit value.



    Args:

        mask_string: A string representing a set of bit masks.



    Returns:

        A dictionary representing the set of bit masks.

    """

    mask_dict = {}

    bit_indices = mask_string.split(',')

    for bit_index in bit_indices:

        try:

            bit_index_int = int(bit_index)

            mask_dict[bit_index_int] = True

        except ValueError:

            continue

    return mask_dict

