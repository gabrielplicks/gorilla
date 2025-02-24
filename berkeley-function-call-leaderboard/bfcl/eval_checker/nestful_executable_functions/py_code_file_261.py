import re

from typing import List



def get_info_from_meta_tag(meta_tag: str, html_doc: str) -> List[str]:

    """Extracts and returns the value of a <meta> tag from a given HTML document.



    Args:

        meta_tag: The name of the <meta> tag to extract.

        html_doc: The HTML document to be searched.



    Returns:

        A list of values of the specified <meta> tags. If there are no <meta> tags with the specified name,

        the function returns an empty list.

    """

    regex = re.compile(r'<meta\s+name="' + re.escape(meta_tag) + r'"\s+content="([^"]*)"')

    matches = re.findall(regex, html_doc)

    return matches

