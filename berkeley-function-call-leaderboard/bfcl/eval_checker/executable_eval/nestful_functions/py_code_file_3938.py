import datetime



def convert_to_minutes(time_string: str) -> int:

    """Converts a time string in the format '%I:%M%p' to a positive integer representing the total number of minutes from midnight.



    Args:

        time_string: The time string to convert.



    Returns:

        The total number of minutes from midnight.

    """

    time_obj = datetime.datetime.strptime(time_string, '%I:%M%p')

    total_minutes = (time_obj.hour * 60 + time_obj.minute) % (24 * 60)



    return total_minutes

