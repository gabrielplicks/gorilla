from typing import List, Tuple



def check_image(image: List[List[Tuple[int, int, int]]], color: Tuple[int, int, int]) -> bool:

    """Checks if any pixel in the image is a specific color.



    Args:

        image: A 2D list representing the image. Each pixel is a tuple of RGB values.

        color: A tuple of RGB values representing the color to check for.



    Returns:

        True if the color is found in the image, False otherwise.

    """

    if not image:

        return False

    for row in image:

        for pixel in row:

            if pixel == color:

                return True



    return False

