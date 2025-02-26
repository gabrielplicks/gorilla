from typing import AnyStr



def extract_content(html: AnyStr) -> str:

    """Extracts the content from a HTML string.



    Args:

        html: The HTML string.



    Returns:

        The content without the HTML tags.

    """

    content = ""



    i = 0

    while i < len(html):

        if html[i] == "<":

            i = html.find(">", i) + 1

            continue



        content += html[i]

        i += 1



    return content

