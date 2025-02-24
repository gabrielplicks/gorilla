import re

from typing import List, Dict



def scrape_data(html_document: str, target_keyword: str) -> List[Dict[str, str]]:

    """Extracts articles containing a target keyword from a given HTML document.



    Args:

        html_document: The HTML document as a string.

        target_keyword: The target keyword to search for in the article titles and contents.



    Returns:

        A list of articles containing the target keyword in the article title or content.

        Each article is represented as a dictionary with the title as the key and the content as the value.

    """

    articles = []

    pattern = re.compile(r'<h1>(.+)</h1><p>(.+)</p>')

    for match in pattern.finditer(html_document):

        title, content = match.groups()

        if target_keyword in title or target_keyword in content:

            articles.append({title: content})

    return articles

