from typing import List, Tuple



def compute_image_dimension(pixel_coordinates: List[Tuple[int, int]], origin: Tuple[int, int]) -> Tuple[int, int]:

    """Computes the dimension of an image given a list of pixel coordinates and the origin pixel location.



    Args:

        pixel_coordinates: A list of pixel coordinates, where each coordinate is a tuple of two integers.

        origin: The origin pixel location, represented as a tuple of two integers.



    Returns:

        A tuple representing the dimension of the image.

    """

    max_x, max_y = 0, 0



    for x, y in pixel_coordinates:

        if x > max_x:

            max_x = x

        if y > max_y:

            max_y = y



    return max_x - origin[0] + 1, max_y - origin[1] + 1

