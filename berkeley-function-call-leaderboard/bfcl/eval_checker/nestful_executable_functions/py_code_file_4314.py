from datetime import datetime



def parse_sk_date(sk_date_str: str) -> datetime:

    """Parses a string in the format "day month year" into a datetime.datetime object.



    Args:

        sk_date_str: The input string in the format "day month year".



    Returns:

        A datetime.datetime object representing the input date.

    """

    format_str = "%d %B %Y"

    parsed_date = datetime.strptime(sk_date_str, format_str)

    return parsed_date

