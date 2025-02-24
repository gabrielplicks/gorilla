import time



def convert_timestamp(timestamp: int) -> str:

    """Converts a Unix timestamp to a date string in the format "YYYY-MM-DD".



    Args:

        timestamp: A Unix timestamp representing a specific point in time.



    Returns:

        A date string in the format "YYYY-MM-DD".

    """

    date = time.gmtime(timestamp)

    date_string = time.strftime("%Y-%m-%d", date)



    return date_string

