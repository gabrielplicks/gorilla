from typing import List



def invert_channels(image: List[List[List[int]]]) -> List[List[List[int]]]:

    """Inverts the channels of an image.



    Args:

        image: A list of pixels, where each pixel is a list of channels.



    Returns:

        A new list of pixels with the channels inverted.

    """

    inverted_image = []

    for pixel in image:

        inverted_pixel = []

        for channel in pixel:

            inverted_channel = channel[::-1]

            inverted_pixel.append(inverted_channel)

        inverted_image.append(inverted_pixel)

    return inverted_image

