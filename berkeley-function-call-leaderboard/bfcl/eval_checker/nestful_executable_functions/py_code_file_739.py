from typing import Tuple



def generate_css_color(hex_color: str) -> str:

    """Generates a CSS color string from a hex color string.



    Args:

        hex_color: A hex color string without the leading hash.



    Returns:

        A CSS color string in the format 'rgb(r,g,b)'.

    """

    r = int(hex_color[:2], base=16)

    g = int(hex_color[2:4], base=16)

    b = int(hex_color[4:6], base=16)



    return f'rgb({r},{g},{b})'

