from typing import Tuple



def highlight(text: str, color: str = 'black', background_color: str = 'transparent') -> str:

    """

    Highlights the given text in the specified color and background color.

    Args:

        text: The input string to be highlighted.

        color: The color of the text. Default is 'black'.

        background_color: The background color of the text. Default is 'transparent'.

    """

    return '<span style="color: {}; background-color: {}">{}</span>'.format(color, background_color, text)

