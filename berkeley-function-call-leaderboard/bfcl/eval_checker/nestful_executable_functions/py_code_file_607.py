from typing import List



def convert_from_srm_to_rgb(srm_colors: List[float]) -> List[tuple]:

    """Converts a list of SRM colors to a list of RGB color tuples.



    Args:

        srm_colors: A list of SRM colors represented as floating point numbers.



    Raises:

        ValueError: If the input list of SRM colors is empty.

    """

    if not srm_colors:

        raise ValueError("No SRM colors to convert.")



    rgb_colors = []



    for srm_color in srm_colors:

        r = srm_color * 2.55

        g = srm_color * 2.55

        b = srm_color * 2.55

        rgb_color = (r, g, b)

        rgb_colors.append(rgb_color)

    return rgb_colors

