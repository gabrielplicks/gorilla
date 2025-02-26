from typing import Tuple



def calculate_total_frames(video_duration: float, frame_rate: float) -> int:

    """Calculates the total number of frames in a video given its duration and frame rate.

    Args:

        video_duration: The duration of the video in seconds.

        frame_rate: The number of frames per second.

    """

    total_frames = video_duration * frame_rate

    return int(total_frames)

