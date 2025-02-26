from typing import Union



def convert_frames_to_milliseconds(no_of_frames: Union[int, float]) -> Union[int, None]:

    """Converts the number of frames (30 frames per second) to the equivalent number of milliseconds.



    Args:

        no_of_frames: The input number of frames.



    Returns:

        An integer representing the equivalent number of milliseconds, or None if the input is invalid.

    """

    if not isinstance(no_of_frames, int) or no_of_frames <= 0:

        print("Error: Invalid input. Input must be a positive, non-zero, non-string integer.")

        return None

    seconds = no_of_frames / 30

    milliseconds = int(seconds * 1000)



    return milliseconds

