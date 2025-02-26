from typing import Union



def encode_time(timestamp: Union[int, float]) -> str:

    """Converts a timestamp in seconds to a string in the format of 'hh:mm:ss'.



    Args:

        timestamp: The timestamp in seconds.

    """

    hours = timestamp // 3600

    minutes = (timestamp % 3600) // 60

    seconds = timestamp % 60

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'



def decode_time(time_str: str) -> int:

    """Converts a string in the format of 'hh:mm:ss' to a timestamp in seconds.



    Args:

        time_str: The string in the format of 'hh:mm:ss'.

    """

    hours, minutes, seconds = map(int, time_str.split(':'))

    return hours * 3600 + minutes * 60 + seconds

