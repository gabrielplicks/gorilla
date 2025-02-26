import re

import typing



def transform_to_text_lines(lines: typing.List[typing.Dict[str, typing.Any]]) -> typing.List[str]:

    """Transforms a list of dictionaries with a required `content` key and an optional `is_output` key to a list of text lines.

    If `is_output` is `True`, then the `content` should be wrapped in parentheses. Otherwise, the `content` should be unmodified.

    Uses a regular expression substitution to transform the `content` of the dictionary.



    Args:

        lines: A list of dictionaries with a required `content` key and an optional `is_output` key.



    Returns:

        A list of text lines.

    """

    return [

        re.sub(r'^(.*)$', r'(\1)' if line.get('is_output', False) else r'\1', line['content'])

        for line in lines

    ]

