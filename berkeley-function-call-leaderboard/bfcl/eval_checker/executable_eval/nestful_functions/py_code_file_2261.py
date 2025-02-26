from typing import List



def is_dirty(hex_view: List[str], offset: int, length: int) -> bool:

    """Checks if a region in a hex view of a file has been modified.



    Args:

        hex_view: A list of hexadecimal strings representing the changes in the file.

        offset: The starting index of the region to be validated.

        length: The length of the region to be validated.

    """

    for i in range(offset, offset + length):

        if hex_view[i] != original_file[i]:

            return True

    return False

