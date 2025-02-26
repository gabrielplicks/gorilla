from typing import Dict, List



def generate_html_links_from_data(links: List[Dict[str, str]]) -> None:

    """Generates HTML links from a list of dictionaries.

    Each dictionary in the list represents a link, with the keys "url" and "text".

    The function yields the HTML link as a string.

    Args:

        links: A list of dictionaries, where each dictionary represents a link.

    """

    for link in links:

        url = link.get("url")

        text = link.get("text")

        html_link = f'<a href="{url}">{text}</a>'

        yield html_link

