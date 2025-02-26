import pandas as pd

from typing import List



def generate_date_list(start_date: str, end_date: str) -> List[bytes]:

    """Generates a list of dates that fall within a given date range.

    Each date is formatted as 'YYYY-MM-DD' and encoded to UTF-8.

    Args:

        start_date: The start date of the range.

        end_date: The end date of the range.

    """

    date_range = pd.date_range(start_date, end_date, freq='D')

    return [date.strftime('%Y-%m-%d').encode('utf-8') for date in date_range]

