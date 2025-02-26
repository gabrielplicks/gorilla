from typing import Union



def device_type(user_agent: str) -> Union[str, None]:

    """Determines the device type based on the user agent string.



    Args:

        user_agent: The user agent string.



    Returns:

        The device type, such as 'mobile', 'tablet', or 'desktop'.

        Returns None if the device type cannot be determined.

    """

    if "Android" in user_agent:

        return "mobile"

    elif "iPhone" in user_agent or "iPad" in user_agent:

        return "tablet"

    elif "Macintosh" in user_agent:

        return "desktop"

    else:

        return None

