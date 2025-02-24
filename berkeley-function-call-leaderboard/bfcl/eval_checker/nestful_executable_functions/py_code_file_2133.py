import re

from typing import Optional



def parse_version_component(version_string: str, component: str) -> Optional[int]:

    """Parses a version string to extract a specific component.



    Args:

        version_string: The version string to parse.

        component: The component to extract (major, minor, or patch).



    Returns:

        The value of the specified component, or None if the component is not found.

    """

    pattern = r'^(\d+).(\d+).(\d+)$'

    match = re.search(pattern, version_string)

    if match:

        major = int(match.group(1))

        minor = int(match.group(2))

        patch = int(match.group(3))

        if component == 'major':

            return major

        elif component == 'minor':

            return minor

        elif component == 'patch':

            return patch

        else:

            return None

    else:

        return None

