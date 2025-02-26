from datetime import datetime

from typing import Dict, Union



def format_page_views(data: Dict[str, Union[int, Dict[str, str], datetime]], date_str: str) -> Union[str, None]:

    """Formats the value of the `page_views` key in the `data` dictionary based on its type.



    Args:

        data: A dictionary containing the data of a website visitor.

        date_str: A string representing the date format.



    Returns:

        The formatted value of the `page_views` key, or None if it's a dictionary.

    """

    page_views = data.get("page_views")

    if isinstance(page_views, int):

        return str(page_views)

    elif isinstance(page_views, dict):

        return None

    elif isinstance(page_views, datetime):

        return page_views.strftime(date_str)

    else:

        raise ValueError("Invalid page_views type")

