from typing import List, Tuple



def average_colors(colors: List[Tuple[int, int, int]]) -> Tuple[float, float, float]:

    """

    Takes a list of RGB tuples and returns a single RGB tuple representing the average color.

    Each RGB tuple contains three integers between 0 and 255, inclusive, representing the

    red, green, and blue components of a color, respectively.

    Args:

        colors: A list of RGB tuples.

    """

    return tuple(sum(channel) / len(colors) for channel in zip(*colors))

