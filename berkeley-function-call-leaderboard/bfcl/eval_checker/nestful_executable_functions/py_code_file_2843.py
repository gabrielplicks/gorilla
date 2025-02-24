from datetime import datetime

from typing import List



def find_max_date(dates: List[str]) -> str:

    """Finds the maximum date in a list of date strings.



    Args:

        dates: A list of date strings in the format `YYYY-MM-DD`.



    Returns:

        The maximum date in the format `YYYY-MM-DD`.

    """

    parsed_dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    parsed_dates.sort()

    return parsed_dates[-1].strftime("%Y-%m-%d")

