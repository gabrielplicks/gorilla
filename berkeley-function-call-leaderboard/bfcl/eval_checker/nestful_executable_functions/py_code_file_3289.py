import locale

from datetime import datetime

from typing import Union



def get_euro_date(date_str: str) -> str:

    """

    Converts a date string from US format to European format.



    Args:

        date_str: The date string in US format.



    Returns:

        The date string in European format.

    """

    locale.setlocale(locale.LC_ALL, '')  # Use the default locale



    date_formats = ['%m/%d/%Y', '%m/%d/%y', '%b %d, %Y', '%b %d, %y']

    for date_format in date_formats:

        try:

            date_obj = datetime.strptime(date_str, date_format)

            return date_obj.strftime('%d.%m.%Y')

        except ValueError:

            continue



    raise ValueError('Invalid date format')

