from typing import Union



def get_number_of_days_in_month(year: int, month: int) -> Union[int, None]:

    """Calculates and returns the number of days in the given month of the given year.

    If the given month is invalid, the function returns None.

    Args:

        year: The given year.

        month: The given month.

    """

    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    month_lengths = [31, 29 if is_leap_year else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if 1 <= month <= 12:

        return month_lengths[month - 1]

    return None

