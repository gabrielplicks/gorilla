from typing import List



def rotate_image_90_degrees(canvas_image: List[List[int]]) -> List[List[int]]:

    """Rotates a canvas image counterclockwise by 90 degrees in-place.



    Args:

        canvas_image: A 2D array representing the canvas image. Each element is an integer representing a pixel color value.



    Returns:

        The rotated canvas image.

    """

    n = len(canvas_image)

    for i in range(n):

        for j in range(i, n):

            canvas_image[i][j], canvas_image[j][i] = canvas_image[j][i], canvas_image[i][j]

    for i in range(n):

        canvas_image[i].reverse()



    return canvas_image

