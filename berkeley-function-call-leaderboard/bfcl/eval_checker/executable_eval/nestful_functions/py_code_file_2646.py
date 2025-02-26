from typing import Dict



def gen_html(attributes: Dict[str, str]) -> str:

    """Generates an HTML element with the given attributes.



    Args:

        attributes: A dictionary where keys represent attribute names and values represent attribute values.

    """

    html = "<div"

    for attribute, value in attributes.items():

        html += f' {attribute}="{value}"'

    html += "></div>"

    return html

