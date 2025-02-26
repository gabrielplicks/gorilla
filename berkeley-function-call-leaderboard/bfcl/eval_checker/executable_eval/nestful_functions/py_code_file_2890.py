from datetime import datetime

from typing import Tuple



def count_days_between_dates(date1: str, date2: str) -> int:

    """Calculates the number of days between two dates.

    Args:

        date1: The first date in the format `dd/mm/yyyy`.

        date2: The second date in the format `dd/mm/yyyy`.

    """

    date1_obj = datetime.strptime(date1, '%d/%m/%Y')

    date2_obj = datetime.strptime(date2, '%d/%m/%Y')

    return (date2_obj - date1_obj).days

