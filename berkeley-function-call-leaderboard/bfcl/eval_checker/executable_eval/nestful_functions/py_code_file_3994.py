from typing import Union



def convert_seconds_to_formatted_string(seconds: Union[int, float]) -> str:

    """Converts a number of seconds into a formatted string of the form `HH:MM:SS`.



    The function pads the hours, minutes, and seconds with leading zeros if necessary

    to ensure the resulting string has exactly two digits per field.



    Args:

        seconds: The number of seconds to convert.



    Returns:

        A formatted string of the form `HH:MM:SS`.

    """

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    seconds = seconds % 60

    return f'{hours:02}:{minutes:02}:{seconds:02}'

