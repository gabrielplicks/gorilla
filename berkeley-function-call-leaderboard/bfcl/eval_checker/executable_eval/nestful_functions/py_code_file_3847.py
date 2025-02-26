import grp

import os

from typing import Union



def check_group(group_name: str) -> Union[bool, None]:

    """Checks if the user running the script belongs to a specific group.

    Args:

        group_name: The name of the group to check.

    Returns:

        True if the user belongs to the group, False if the group doesn't exist or the user doesn't belong to the group.

    """

    try:

        group = grp.getgrnam(group_name)

        current_user_gid = os.getgid()

        if current_user_gid == group.gr_gid:

            return True

        else:

            return False

    except KeyError:

        return False

