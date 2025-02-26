from typing import Union



def convert_to_HHMMSS(t: Union[int, float]) -> str:

    """Converts a non-negative integer or float `t` representing a duration in seconds into a string with the format `HH:MM:SS`, where `HH` means hours, `MM` means minutes, and `SS` means seconds.



    Args:

        t: The duration in seconds.



    Returns:

        A string with the format `HH:MM:SS`.

    """

    hours = t // 3600

    minutes = (t % 3600) // 60

    seconds = (t % 3600) % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

