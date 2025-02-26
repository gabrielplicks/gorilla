import re



def extract_time(time_string: str) -> tuple:

    """Extracts the hours, minutes, and seconds from a string containing a time duration in the format "Xh Ym Zs".



    Args:

        time_string: A string containing a time duration in the format "Xh Ym Zs".



    Returns:

        A tuple of integers representing the hours, minutes, and seconds.



    Raises:

        ValueError: If the input string does not match the expected format.

    """

    pattern = r"(\d+)h (\d+)m (\d+)s"

    match = re.match(pattern, time_string)

    if match:

        hours, minutes, seconds = match.groups()

        return (int(hours), int(minutes), int(seconds))

    else:

        raise ValueError("Invalid time format")

