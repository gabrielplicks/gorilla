from typing import Set, Literal



def is_valid_css_keyword(keyword: str) -> bool:

    """Checks if a given string is a valid CSS keyword.



    Args:

        keyword: The string to check.



    Returns:

        True if the keyword is valid, False otherwise.

    """

    valid_keywords: Set[Literal['initial', 'inherit', 'unset', 'revert', 'auto', 'normal', 'revert-layer', 'revert-layer-block', 'revert-layer-inline']] = {'initial', 'inherit', 'unset', 'revert', 'auto', 'normal', 'revert-layer', 'revert-layer-block', 'revert-layer-inline'}

    return keyword in valid_keywords

