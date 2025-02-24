from typing import Union



def capitalize_time_zone(name: Union[str, bytes]) -> str:

    """

    Capitalize the first letter of each word in a time zone name.

    Args:

        name: The time zone name.

    """

    return name.title()

