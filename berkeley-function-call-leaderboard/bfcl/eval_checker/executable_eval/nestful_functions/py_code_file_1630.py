from typing import List



def format_version_number(version_number: str) -> str:

    """Formats a version number string by splitting it into a list of strings representing each number,

    then iterating through the list to create a new list of strings where each string is a formatted number,

    with a comma separating each number. Finally, joins the new list of strings into a single string and returns it.

    Args:

        version_number: A version number string like '3.14.1592'.

    """

    number_list: List[str] = version_number.split(".")

    formatted_list: List[str] = [f"{num:03d}" for num in map(int, number_list)]

    formatted_version_number: str = ".".join(formatted_list)

    return formatted_version_number

