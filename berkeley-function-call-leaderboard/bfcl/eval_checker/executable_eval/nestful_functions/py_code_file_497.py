import datetime



def get_year(date: str) -> int:

    """Extracts the year from a date string in the format 'DD/MM/YYYY'.



    Args:

        date: The date string in the format 'DD/MM/YYYY'.



    Returns:

        The year component of the date.

    """

    parsed_date = datetime.datetime.strptime(date, '%d/%m/%Y')

    year = parsed_date.year

    return year

