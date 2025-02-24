from typing import Tuple



def extract_element_text(html: str, element: str) -> str:

    """Extracts the text contents of a specific element from a given HTML string.



    Args:

        html: A string containing the HTML code.

        element: A string representing the element tag to extract.



    Returns:

        The text contents of the specified element, or a message indicating that the element was not found.

    """

    start_position = html.find("<" + element + ">")

    end_position = html.find("</" + element + ">")



    if start_position == -1 or end_position == -1:

        return "No such element found."



    return html[start_position + len(element) + 2:end_position]

