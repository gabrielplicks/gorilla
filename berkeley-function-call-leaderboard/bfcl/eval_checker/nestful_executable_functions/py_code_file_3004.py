from datetime import datetime



def get_formatted_time_string(dt: datetime) -> str:

    """Returns a string with the format 'YYYY-MM-DD hh:mm:ss' from a datetime object.



    Args:

        dt: The datetime object.



    Returns:

        A string with the format 'YYYY-MM-DD hh:mm:ss'.

    """

    return dt.strftime('%Y-%m-%d %H:%M:%S')

