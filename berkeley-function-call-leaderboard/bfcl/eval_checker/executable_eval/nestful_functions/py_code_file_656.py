from typing import Tuple



def day_of_year_to_month_day(day_of_year: int) -> Tuple[int, int]:

    """Converts the day of the year to a month and day.



    Args:

        day_of_year: The day of the year as an integer.



    Returns:

        A tuple containing the month and day.

    """

    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    day = day_of_year

    month = 1

    if day > 59:

        month_lengths[1] = 29  # Handle leap years by considering February 29th as the 60th day

    for month_length in month_lengths:

        if day <= month_length:

            return month, day

        month += 1

        day -= month_length

