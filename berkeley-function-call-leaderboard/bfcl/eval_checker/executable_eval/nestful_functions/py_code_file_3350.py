from typing import Tuple



def timecode_to_frame_number(timecode: str, frame_rate: float) -> int:

    """Converts a timecode to the equivalent frame number given a frame rate.



    Args:

        timecode: The timecode in the format `HH:MM:SS:FF`.

        frame_rate: The frame rate.



    Returns:

        The total frame number.

    """

    hours, minutes, seconds, frames = map(int, timecode.split(":"))

    total_seconds = hours * 60 * 60 + minutes * 60 + seconds

    return total_seconds * frame_rate + frames

