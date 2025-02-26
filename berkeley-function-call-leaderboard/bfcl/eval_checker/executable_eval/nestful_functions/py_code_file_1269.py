from datetime import datetime

from typing import Union



def parse_and_format(string_to_parse: Union[str, bytes]) -> str:

    """Parses a string into a datetime object and returns it in ISO 8601 format.



    Args:

        string_to_parse: The string to parse.



    Returns:

        The string in ISO 8601 format.

    """

    date_tuple = datetime.strptime(string_to_parse, '%Y-%m-%dT%H:%M:%S')

    date_object = datetime(date_tuple.year, date_tuple.month, date_tuple.day, date_tuple.hour, date_tuple.minute, date_tuple.second)

    iso_format = date_object.isoformat()

    return iso_format

