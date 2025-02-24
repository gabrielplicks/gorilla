from typing import List



def flip_image(image: List[List[int]]) -> List[List[int]]:

    """Flips an image horizontally (i.e., left-to-right).



    Args:

        image: A list of lists of numbers representing an image.



    Returns:

        A flipped image.

    """

    return [list(reversed(row)) for row in image]

