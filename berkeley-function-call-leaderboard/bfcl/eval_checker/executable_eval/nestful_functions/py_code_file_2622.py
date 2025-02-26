from typing import List



def create_bulleted_list(items: List[str], bold: bool = False) -> str:

    """Creates a bulleted list from a list of strings.

    Args:

        items: A list of strings to be formatted as a bulleted list.

        bold: A boolean indicating whether the items should be bolded.

    """

    bullet_items = []

    for item in items:

        bullet_item = f"<li>{item}</li>"

        if bold:

            bullet_item = f"<li><b>{item}</b></li>"

        bullet_items.append(bullet_item)

    return f"<ul>{''.join(bullet_items)}</ul>"

