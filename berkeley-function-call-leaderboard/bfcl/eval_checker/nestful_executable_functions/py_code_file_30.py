from typing import List



def generate_palette(hue: int, shades: List[float]) -> List[str]:

    """Generates a palette of colors for a given theme with a specific hue and a set of shades.

    Args:

        hue: The hue value in the range of 0 to 360.

        shades: A set of shades such as "light", "medium", "dark", or "very dark".

    """

    palette = []

    for shade in shades:

        palette.append(f"HSV({round(hue, -1)}, 100%, {round(shade * 10, -1)}%)")

    return palette

