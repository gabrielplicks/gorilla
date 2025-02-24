import re

from typing import List



def extract_images_from_html(html_string: str) -> List[str]:

    """Extracts the URLs of all images referenced in an HTML string.



    Args:

        html_string: The HTML string to extract image URLs from.



    Returns:

        A list of the URLs of all images referenced in the HTML string.

    """

    pattern = r'src="(.+?)"'  # Matching src attribute

    return re.findall(pattern, html_string)

