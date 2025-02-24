from typing import Union

import math



def calculate_frame_number(time_in_seconds: Union[int, float], fps: Union[int, float]) -> int:

    """Calculates the frame number at a given time based on a constant frame rate (FPS).

    Args:

        time_in_seconds: The time in seconds.

        fps: The frame rate in frames per second.

    """

    return round(time_in_seconds * fps)

