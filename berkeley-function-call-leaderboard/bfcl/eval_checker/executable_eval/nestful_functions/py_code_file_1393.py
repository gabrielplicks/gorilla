import re



def extract_html_tag_names(html: str) -> list[str]:

    """Extracts all HTML tag names from the input string.



    Args:

        html: The input string of HTML code.



    Returns:

        A list of all HTML tag names found within the code, without their angle brackets or attributes.

    """

    return re.findall(r'<([a-zA-Z]+)[^>]*>', html)

