import random

random.seed(42)
from typing import List, Union



def get_frames(frames: List[Union[int, float]], probability: float) -> Union[List[Union[int, float]], str]:

    """

    Returns a list of frames sampled randomly with a specified probability.

    If the input list is empty, returns a warning message.



    Args:

        frames: A list of frames.

        probability: The probability of sampling a frame.

    """

    if not frames:

        return "Warning: Input list is empty."



    output_frames = []

    for frame in frames:

        random_number = random.random()

        if random_number < probability:

            output_frames.append(frame)



    return output_frames

