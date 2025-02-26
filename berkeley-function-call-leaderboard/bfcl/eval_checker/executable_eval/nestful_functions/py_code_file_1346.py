from typing import Union



def convert_seconds(sec: int) -> str:

    """Converts an integer `sec` (representing the number of seconds) to a string in the format of 'HH:MM:SS'.



    Args:

        sec: The number of seconds to convert.



    Returns:

        A string in the format of 'HH:MM:SS'.

    """

    hours = sec // 3600

    minutes = (sec % 3600) // 60

    seconds = sec % 60

    return f'{hours:02}:{minutes:02}:{seconds:02}'

