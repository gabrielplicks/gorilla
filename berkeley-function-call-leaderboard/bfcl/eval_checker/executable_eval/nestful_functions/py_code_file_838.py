import re

import typing



def concatenate_html_strings(html_strings: typing.List[str]) -> str:

    """Concatenates a list of HTML strings into a single string with no HTML tags.



    Args:

        html_strings: A list of HTML strings.



    Returns:

        A single string that is a concatenation of the input strings and has no HTML tags embedded within it.

    """

    result = ""



    for html_string in html_strings:

        text_content = re.sub(r"<[^>]*>", "", html_string)

        result += text_content



    return result

