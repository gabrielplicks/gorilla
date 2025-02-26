import re

from typing import List



def extract_usernames(messages: List[str]) -> List[str]:

    """Extracts usernames from a list of messages.



    Args:

        messages: A list of messages that contain usernames in the format "User: Username".



    Returns:

        A list of usernames extracted from the messages.

    """

    usernames = []

    pattern = r"User: (\w+)"  # Regular expression pattern to match "User: Username"



    for message in messages:

        match = re.match(pattern, message)

        if match:

            username = match.group(1)  # Extract the first capture group

            usernames.append(username)



    return usernames

