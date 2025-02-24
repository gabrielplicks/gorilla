from datetime import datetime, timedelta

import re



def process_date(date: str, days: int) -> str:

    """

    Calculates a new date that is `days` days after the input date.



    Args:

        date: The input date in `YYYY-MM-DD` format.

        days: The number of days to add to the input date.



    Raises:

        ValueError: If the input date is invalid.

    """

    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):

        raise ValueError("Invalid date format")

    year, month, day = map(int, date.split("-"))

    date_obj = datetime(year, month, day)

    new_date_obj = date_obj + timedelta(days=days)

    new_date = new_date_obj.strftime("%Y-%m-%d")



    return new_date

