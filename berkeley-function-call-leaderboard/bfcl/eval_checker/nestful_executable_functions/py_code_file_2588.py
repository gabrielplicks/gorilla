from typing import List



def get_mask(arr: List[str], mask: str) -> List[str]:

    """

    Returns a new array of strings that only includes elements from the original array

    that are masked by the string parameter.

    Args:

        arr: The original array of strings.

        mask: The mask string.

    """

    result = []

    mask_bin = bin(int(mask, 2))[2:]

    for el in arr:

        if el is None:

            el_bin = "0" * len(mask_bin)

        else:

            el_bin = bin(int(el, 2))[2:]

        if len(mask_bin) == len(el_bin):

            if mask_bin == el_bin:

                result.append(el)

    return result

