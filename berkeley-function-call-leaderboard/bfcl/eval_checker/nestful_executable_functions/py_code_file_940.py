import re

from typing import Tuple



def generate_font_css(font_family: str, subset: str) -> str:

    """Generates a CSS stylesheet for a specific font using a specified character subset.



    Args:

        font_family: The name of the font to be used.

        subset: A string of characters to be included in the font.



    Returns:

        A CSS stylesheet as a string that contains `@font-face` rules for the font family and its subset.

    """

    font_family = re.sub(r'([^\w\s])', '\\\1', font_family)

    style_rule = f"""@font-face {{

    font-family: "{font_family}";

    src: url("{font_family}-{subset}.woff2") format("woff2");

}}"""



    return style_rule

