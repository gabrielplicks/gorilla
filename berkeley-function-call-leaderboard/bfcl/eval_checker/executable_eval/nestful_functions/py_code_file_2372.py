from typing import List



def convert_time_to_list(time: str) -> List[int]:

    """Converts a string containing a time in the format `"HH:MM:SS"` to the corresponding digital time in units of hours, minutes, and seconds (all integers).



    Args:

        time: A string containing a time in the format `"HH:MM:SS"`.



    Returns:

        A list of integers representing the hours, minutes, and seconds.

    """

    hours, minutes, seconds = time.split(':')

    hours = int(hours)

    minutes = int(minutes)

    seconds = int(seconds)

    return [hours, minutes, seconds]

