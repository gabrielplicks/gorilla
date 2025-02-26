import datetime



def convert_seconds_to_datestring(seconds: int) -> str:

    """

    Converts a time duration in seconds to a human-readable string.



    Args:

        seconds: The time duration in seconds.



    Returns:

        A string in the format "YYYY-MM-DD hh:mm:ss" representing the time duration.

    """

    time_delta = datetime.timedelta(seconds=seconds)

    timestamp = datetime.datetime(1970, 1, 1, 0, 0, 0) + time_delta

    datestring = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return datestring

