import pytz



def timezone_exists(tz_string: str) -> bool:

    """Checks if a given timezone string exists in the list of known timezones.



    Args:

        tz_string: The timezone string to check.



    Returns:

        True if the timezone exists, False otherwise.

    """

    return tz_string in pytz.all_timezones

