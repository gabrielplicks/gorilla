from typing import Union



def seconds_to_time_string(seconds: Union[int, float]) -> str:

    """Converts the number of seconds into a human-readable string of the format `HH:MM:SS`.



    Args:

        seconds: The number of seconds to convert.



    Returns:

        A string of the format `HH:MM:SS`.

    """

    hours = int(seconds // 3600)

    minutes = int((seconds % 3600) // 60)

    seconds = int(seconds % 60)

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

