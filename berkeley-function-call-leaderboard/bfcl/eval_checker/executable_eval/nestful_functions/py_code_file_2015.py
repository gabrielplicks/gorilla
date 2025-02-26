import re

from typing import List



def extract_text_from_a_tags(html: str) -> List[str]:

    """Extracts the text content from all `<a>` (anchor) elements in an HTML string.



    Excludes text from `<script>` and `<style>` elements.



    Args:

        html: The HTML string to extract text from.



    Returns:

        A list of strings, where each string is the text content of an `<a>` tag.

    """

    html = re.sub(r'<(script|style).*?</\1>', '', html, flags=re.DOTALL)

    result = re.findall(r'<a[^>]*>(.*?)</a>', html)

    return result

