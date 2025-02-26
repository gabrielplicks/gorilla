from typing import List



def speed_index(timestamps: List[float], frame_sizes: List[float]) -> float:

    """Calculates the speed index based on the given timestamps and frame sizes.



    The speed index is a weighted average of the perceptual speed of the video,

    where each frame size is associated with a specific timestamp.



    Args:

        timestamps: A list of timestamps sorted in ascending order.

        frame_sizes: A list of corresponding frame sizes.



    Returns:

        The speed index as a weighted average.

    """

    weighted_average = 0

    for timestamp, frame_size in zip(timestamps, frame_sizes):

        perceptual_speed = 1 / timestamp

        weighted_average += frame_size * perceptual_speed

    return weighted_average

