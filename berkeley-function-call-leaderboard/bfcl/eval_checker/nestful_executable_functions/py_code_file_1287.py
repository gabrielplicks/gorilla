from typing import Dict



def get_style_name(style: Dict[str, str]) -> str:

    """Returns the style name and family name as a string separated by a space.

    If the style object doesn't have a `style_name` key, returns `None`.

    Args:

        style: A dictionary containing the style information.

    """

    try:

        style_name = style['style_name']

    except KeyError:

        return None



    if 'family' in style:

        return f"{style_name} {style['family']}"

    return style_name

