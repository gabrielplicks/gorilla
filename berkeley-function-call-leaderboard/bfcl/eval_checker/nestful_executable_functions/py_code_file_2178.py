from typing import List



def convert_to_regex(strings: List[str], regex_group_name: str) -> str:

    """Converts a list of strings into a regular expression that matches any of the strings in the list.

    Args:

        strings: A list of strings.

        regex_group_name: The name of the group that will capture the string.

    """

    regex = f"(?P<{regex_group_name}>" + ")|(".join(strings) + ")"

    return regex

