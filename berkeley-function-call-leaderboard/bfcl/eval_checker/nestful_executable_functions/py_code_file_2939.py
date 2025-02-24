from typing import Dict



def get_video_location(video: Dict[str, str]) -> str:

    """Returns the location field of a video object if it exists; otherwise, returns an empty string.



    Args:

        video: A dictionary representing a video object.



    Returns:

        The location field of the video object if it exists; otherwise, an empty string.

    """

    if 'location' in video:

        return video['location']

    else:

        return ""

