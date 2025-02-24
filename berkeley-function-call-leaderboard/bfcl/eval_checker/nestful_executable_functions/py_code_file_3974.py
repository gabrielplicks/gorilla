from typing import Union



def duration_string(duration: Union[int, float]) -> str:

    """Returns a string formatted as `{h}h {m}m {s}s`, where `{h}` is the number of hours in the duration, `{m}` is the number of minutes in the duration, and `{s}` is the number of seconds in the duration. The function returns `0s` if the duration is 0 or less.



    Args:

        duration: The duration in seconds.

    """

    if duration <= 0:

        return "0s"

    hours, remainder = divmod(duration, 3600)

    minutes, seconds = divmod(remainder, 60)

    return f"{hours}h {minutes}m {seconds}s"

