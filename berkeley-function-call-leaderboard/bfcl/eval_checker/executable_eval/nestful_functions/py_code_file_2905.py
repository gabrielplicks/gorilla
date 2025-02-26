from datetime import datetime

from typing import List, Union



def str_to_datetime(time_strings: List[str]) -> List[Union[datetime, None]]:

    """Converts a list of time strings to a list of Python datetime objects.

    The time strings may be in different formats. The function handles invalid time strings and

    raises an error for them.

    Args:

        time_strings: A list of time strings.

    Returns:

        A list of datetime objects or None for invalid time strings.

    """

    datetime_objects = []



    for time_string in time_strings:

        try:

            datetime_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')

            datetime_objects.append(datetime_object)

        except ValueError:

            datetime_objects.append(None)



    return datetime_objects

