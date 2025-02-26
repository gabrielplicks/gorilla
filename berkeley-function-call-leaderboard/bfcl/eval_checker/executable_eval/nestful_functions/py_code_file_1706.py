from typing import Dict



def generate_meta_tags(data: Dict[str, str]) -> str:

    """Generates a set of HTML meta tags from a dictionary of name-value pairs.

    Args:

        data: A dictionary of name-value pairs.

    """

    html = ''

    for key, value in data.items():

        html += '<meta name="{}" content="{}">\n'.format(key, value)

    return html

