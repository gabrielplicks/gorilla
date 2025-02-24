from datetime import datetime

from typing import Union



def convert_to_exif_format(date_time_string: Union[str, datetime]) -> str:

    """Converts a date and time string into the EXIF standard format.

    Args:

        date_time_string: A date and time string in the "YYYY-MM-DD HH:MM:SS" format, or a datetime object.

    """

    if isinstance(date_time_string, str):

        date_time_object = datetime.strptime(date_time_string, "%Y-%m-%d %H:%M:%S")

    else:

        date_time_object = date_time_string

    return date_time_object.strftime("%Y:%m:%d %H:%M:%S")

