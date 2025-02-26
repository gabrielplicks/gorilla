import random

random.seed(42)
import typing



def generate_random_rgb_color() -> typing.Tuple[int, int, int]:

    """Generates a random RGB color in a tuple format.

    The RGB color space ranges from 0 to 255.

    The output format is a tuple of three values in the order of (red, green, blue).

    Returns:

        A tuple of three random numbers in the range of 0 to 255.

    """

    red = random.randint(0, 255)

    green = random.randint(0, 255)

    blue = random.randint(0, 255)

    return (red, green, blue)

