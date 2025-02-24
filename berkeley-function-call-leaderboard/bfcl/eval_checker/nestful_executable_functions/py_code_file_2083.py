from typing import Tuple



def create_link(label: str, url: str) -> str:

    """Creates a hyperlink in HTML given a label and a URL.



    Args:

        label: The label for the hyperlink.

        url: The URL for the hyperlink.



    Returns:

        An HTML string with the `<a>` tag containing the link.

    """

    escaped_label = label.replace("\"", "&quot;").replace("'", "&apos;")

    escaped_url = url.replace("\"", "&quot;").replace("'", "&apos;")

    return f'<a href="{escaped_url}">{escaped_label}</a>'

