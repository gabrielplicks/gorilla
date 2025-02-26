from typing import List



def rotate_90_ccw(image: List[List[int]]) -> List[List[int]]:

    """Rotates a list of lists representing an image 90 degrees counter-clockwise.



    Args:

        image: A list of lists containing the values of each pixel in an image.



    Returns:

        The rotated image as a list of lists.

    """

    transposed_image = [list(row) for row in zip(*image)]

    rotated_image = [row[::-1] for row in transposed_image]



    return rotated_image

