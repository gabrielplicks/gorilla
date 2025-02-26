import datetime

from typing import List



def check_dates_for_increase_by_one_day(date_strings: List[str]) -> bool:

    """Checks if consecutive dates in a list differ by no more than one day.



    Args:

        date_strings: A list of date strings in the format '%Y-%m-%d %H:%M:%S'.



    Returns:

        True if the difference between consecutive dates is less than one day, False otherwise.

    """

    for i in range(len(date_strings) - 1):

        date1 = datetime.datetime.strptime(date_strings[i], '%Y-%m-%d %H:%M:%S')

        date2 = datetime.datetime.strptime(date_strings[i+1], '%Y-%m-%d %H:%M:%S')

        difference = abs(date2 - date1)

        if difference > datetime.timedelta(days=1):

            return False

    return True

