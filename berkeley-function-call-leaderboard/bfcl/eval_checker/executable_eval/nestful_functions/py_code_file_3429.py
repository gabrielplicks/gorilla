import random

random.seed(42)
import typing



def generate_colors(n: int = 16) -> typing.List[int]:

    """Generates a list of RGB colors based on the number of colors specified.



    Args:

        n: The number of colors to generate. Default is 16.



    Returns:

        A list of integers where each integer represents a color in the RGB color space.

    """

    colors = []

    for _ in range(n):

        red = random.getrandbits(1)

        green = random.getrandbits(1)

        blue = random.getrandbits(1)

        color = red | (green << 8) | (blue << 16)

        colors.append(color)

    return colors

