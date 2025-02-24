import re

from typing import Pattern, Match



def map_url_prefix(path: str) -> str:

    """Maps a URL path to a URL prefix.



    Args:

        path: The URL path to map.



    Returns:

        The mapped URL prefix.



    Raises:

        Exception: If the path is invalid.

    """

    pattern: Pattern[str] = re.compile(r"(?P<prefix>/api/(?P<version>\d+\.\d+))(?P<suffix>/.*)")

    match: Match[str] = pattern.search(path)

    if match:

        return match.group("prefix")

    else:

        raise Exception("Invalid path")

