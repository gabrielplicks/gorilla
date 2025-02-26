from datetime import datetime



def convert_datetime_to_string(dt: datetime) -> str:

    """Converts a datetime object to a string representing the corresponding date and time in the format "YYYY-MM-DD HH:MM:SS".



    Args:

        dt: The datetime object to be converted.



    Returns:

        A string representing the corresponding date and time in the format "YYYY-MM-DD HH:MM:SS".

    """

    return dt.strftime('%Y-%m-%d %H:%M:%S')

