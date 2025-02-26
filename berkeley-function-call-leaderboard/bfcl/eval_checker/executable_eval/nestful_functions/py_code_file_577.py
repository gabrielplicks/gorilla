from typing import List, Tuple



def format_as_html(data: List[Tuple[str, str]]) -> str:

    """Formats a list of tuples as HTML table rows.



    Args:

        data: A list of tuples, where each tuple contains a key and a value.



    Returns:

        A string formatted as HTML table rows, where for each tuple the key is used as the table header and the value as table data.

    """

    html_rows = ['<tr><th>{}</th><td>{}</td></tr>'.format(key, value) for key, value in data]

    html = ''.join(html_rows)

    return html

