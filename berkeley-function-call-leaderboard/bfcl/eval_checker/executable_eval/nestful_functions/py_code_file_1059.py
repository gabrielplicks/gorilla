import calendar



def get_num_days(month: int, year: int) -> int:

    """Given a month and year, return the number of days in that month/year pair.

    If the month or year is invalid, return -1.

    Args:

        month: The month as an integer (1-12).

        year: The year as a positive integer.

    """

    if not (1 <= month <= 12 and year > 0):  # Check if the month and year are valid

        return -1

    return calendar.monthrange(year, month)[1]  # Get the number of days in the given month

