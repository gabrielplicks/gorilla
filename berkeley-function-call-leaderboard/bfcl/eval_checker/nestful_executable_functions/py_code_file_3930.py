from typing import List, Tuple



def clip_image(image: List[List[int]], min: int, max: int) -> List[List[int]]:

    """Clips the RGB values of each pixel in an image to be within a given range.



    Args:

        image: A 2D list representing an image, where each inner list is a pixel with RGB values.

        min: The minimum value for the RGB values.

        max: The maximum value for the RGB values.

    """

    for pixel in image:

        for i in range(len(pixel)):

            if pixel[i] < min:

                pixel[i] = min

            if pixel[i] > max:

                pixel[i] = max



    return image

