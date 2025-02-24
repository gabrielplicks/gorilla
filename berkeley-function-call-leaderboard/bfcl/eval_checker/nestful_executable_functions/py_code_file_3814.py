from typing import List, Union



def text_center(strings: Union[str, List[str]]) -> str:

    """Centers any number of strings on the screen, separated by a space.

    Args:

        strings: The strings to be centered.

    """

    if isinstance(strings, str):

        strings = [strings]

    max_len = max(len(s) for s in strings)

    width = max(max_len, 30)

    fill_char = ' '

    centered_strings = [s.center(width, fill_char) for s in strings]

    return '\n'.join(centered_strings)

