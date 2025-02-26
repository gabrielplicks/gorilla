import random

random.seed(42)
from typing import Tuple



def random_color() -> Tuple[int, int, int]:

    """Generates a random color in RGB format.



    Returns:

        A tuple of three integers representing the red, green, and blue (RGB) components of the color.

        Each component is an integer ranging from 0 to 255.

    """

    red = random.randint(0, 255)

    green = random.randint(0, 255)

    blue = random.randint(0, 255)

    return (red, green, blue)

