from datetime import datetime, timedelta



def offset_datetime(date_string: str, offset_seconds: int) -> datetime:

    """

    Applies a time offset to a given date and returns a new datetime object.



    Args:

        date_string: The input date in the format "YYYY-MM-DD HH:MM:SS".

        offset_seconds: The time offset in seconds.



    Raises:

        ValueError: If the input date is invalid.

    """

    try:

        input_datetime = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")

        offset = timedelta(seconds=offset_seconds)

        result_datetime = input_datetime + offset

        return result_datetime



    except ValueError:

        raise ValueError("Invalid date or time format.")

