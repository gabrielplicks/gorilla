from typing import Dict



def is_request_for_mp3(headers: Dict[str, str]) -> bool:

    """Checks if the request is for a media type of "audio/mpeg".



    Args:

        headers: A dictionary of request headers.



    Returns:

        A boolean indicating whether the request is for a media type of "audio/mpeg".

    """

    if "Accept" in headers and "audio/mpeg" in headers["Accept"]:

        return True

    else:

        return False

