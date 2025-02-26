from typing import Dict, List, Any



class Video:

    def __init__(self, id: str, title: str, duration: int):

        self.id = id

        self.title = title

        self.duration = duration



def convert_video_list_to_dict(video_list: List[Video]) -> Dict[str, Video]:

    """Converts a list of video objects to a dictionary, where the keys are the video IDs and the values are the video objects.



    Args:

        video_list: A list of video objects.



    Returns:

        A dictionary where the keys are the video IDs and the values are the video objects.

    """

    video_dict = {}

    for video in video_list:

        video_dict[video.id] = video

    return video_dict

