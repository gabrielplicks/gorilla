import datetime



def check_date_format(date_string: str) -> datetime.datetime:

    """Checks if the input date string is in the format YYYY-mm-dd and returns the parsed date object.



    Args:

        date_string: The input date string to be checked.



    Returns:

        The parsed date object.



    Raises:

        ValueError: If the input date string is not in the format YYYY-mm-dd.

    """

    try:

        date = datetime.datetime.strptime(date_string, "%Y-%m-%d")

        return date

    except ValueError:

        raise ValueError("Invalid date format")

