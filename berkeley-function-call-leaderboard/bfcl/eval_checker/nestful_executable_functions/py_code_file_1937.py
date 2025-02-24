from datetime import datetime

from typing import List



def num_days_between(dates: List[str]) -> int:

    """Calculates the number of days between the earliest and latest dates in a list of dates.



    Args:

        dates: A list of dates in string format (YYYY-MM-DD).



    Returns:

        The number of days between the earliest and latest dates in the list.

    """

    datetime_dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    date_range = max(datetime_dates) - min(datetime_dates)

    return date_range.days

