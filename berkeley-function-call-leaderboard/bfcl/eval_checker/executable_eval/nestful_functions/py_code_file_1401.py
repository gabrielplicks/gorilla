from typing import Union



def is_mobile(user_agent: str) -> bool:

    """Determines whether a user agent string is from a mobile device or not.



    Args:

        user_agent: The user agent string to check.



    Returns:

        True if the user agent string contains the substring 'Mobile' or 'Android', and False otherwise.

    """

    return 'Mobile' in user_agent or 'Android' in user_agent

