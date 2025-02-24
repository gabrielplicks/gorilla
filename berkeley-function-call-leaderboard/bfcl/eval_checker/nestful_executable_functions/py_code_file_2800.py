from datetime import datetime

from typing import Optional



def serialize_datetime(datetime_object: Optional[datetime]) -> str:

    """Serializes a datetime object as a string in the format `YYYY-MM-DDTHH:MM:SS`.

    If the input is `None`, the function returns an empty string.



    Args:

        datetime_object: The datetime object to be serialized.

    """

    if datetime_object is None:

        return ''



    return datetime_object.strftime('%Y-%m-%dT%H:%M:%S')

