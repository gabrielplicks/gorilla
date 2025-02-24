from typing import Tuple



def extract_pixel_position(flattened_array_index: int) -> Tuple[int, int]:

    """Extracts the pixel position (x, y) from the flattened array index.

    Args:

        flattened_array_index: The index of the pixel in the flattened array.

    """

    width = 10

    x = flattened_array_index % width

    y = flattened_array_index // width

    return x, y

