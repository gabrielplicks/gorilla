import grp

from typing import Union



def gid2name(gid: int) -> Union[str, None]:

    """Converts a numeric group ID to its corresponding group name.

    If the group ID is not found, then the function returns the group ID as a string.

    Also, handles any potential `KeyError` exceptions gracefully and returns `None` in that case.

    Args:

        gid: The numeric group ID.

    """

    try:

        group_info = grp.getgrgid(gid)

        return group_info.gr_name

    except KeyError:

        return str(gid)

