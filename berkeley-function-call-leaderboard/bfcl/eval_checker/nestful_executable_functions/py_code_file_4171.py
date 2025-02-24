from typing import List



def extract_and_reverse(sequence: List[Any], frame1: int, frame2: int) -> List[Any]:

    """Extracts the frames with indices `frame1` and `frame2` from the sequence and returns a new sequence of frames with the extracted frames in reverse order.



    Args:

        sequence: The original sequence of frames.

        frame1: The index of the first frame to extract.

        frame2: The index of the second frame to extract.



    Returns:

        A new sequence of frames with the extracted frames in reverse order.

    """

    extracted_frames = [sequence[frame1], sequence[frame2]]

    reversed_frames = list(reversed(extracted_frames))

    return reversed_frames

