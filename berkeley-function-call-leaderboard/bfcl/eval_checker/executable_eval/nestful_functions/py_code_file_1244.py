import re



def remove_timezone(datetime_string: str) -> str:

    """Removes the time zone from a datetime string in the format of `YYYY-MM-DDTHH:MM:SS-HH:00`.



    If the input string does not have a time zone, then raise a `ValueError` exception.



    Args:

        datetime_string: The datetime string to remove the time zone from.



    Raises:

        ValueError: If the input string does not have a time zone.

    """

    match = re.search(r'-(?P<tz>\d{2}:\d{2})$', datetime_string)

    if match:

        tz = match.group('tz')

        datetime_string = datetime_string.replace(f'-{tz}', '')

    else:

        raise ValueError('No time zone found')



    return datetime_string

