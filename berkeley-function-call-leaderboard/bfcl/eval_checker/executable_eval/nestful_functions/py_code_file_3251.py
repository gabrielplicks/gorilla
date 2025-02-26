from typing import List



def threshold_image(image: List[List[int]]) -> List[List[int]]:

    """Applies a basic thresholding algorithm to classify each pixel as black or white.

    If the pixel value is less than or equal to 127, assign it to 0 (black).

    If the pixel value is greater than 127, assign it to 255 (white).

    Returns the new 2D array with the thresholded pixel values.



    Args:

        image: A 2D array of pixel colors represented as an integer between 0 and 255.

    """

    thresholded_image = [[0 for _ in row] for row in image]

    for i, row in enumerate(image):

        for j, pixel in enumerate(row):

            if pixel <= 127:

                thresholded_image[i][j] = 0

            else:

                thresholded_image[i][j] = 255



    return thresholded_image

