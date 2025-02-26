import re



def add_paragraph_before_h3(html_content: str) -> str:

    """Adds a paragraph tag before each h3 tag in the given HTML content.



    Args:

        html_content: The HTML content to modify.



    Returns:

        The modified HTML content.

    """

    pattern = re.compile(r'<h3>(.*?)</h3>')

    h3_tags = pattern.findall(html_content)

    for h3_tag in h3_tags:

        html_content = html_content.replace(f'<h3>{h3_tag}</h3>', f'<p>Paragraph before h3 tag</p><h3>{h3_tag}</h3>')



    return html_content

