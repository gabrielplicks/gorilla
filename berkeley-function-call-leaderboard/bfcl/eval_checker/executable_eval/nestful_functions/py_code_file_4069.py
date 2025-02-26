import re

import typing



def extract_chinese_characters(text: str) -> typing.List[str]:

    """Extracts Chinese characters from a string using regular expressions.



    Args:

        text: The input string.



    Returns:

        A list of Chinese characters extracted from the input string.

    """

    assert isinstance(text, str)

    return re.findall(r'[\u4e00-\u9fff]+', text)

