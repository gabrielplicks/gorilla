from typing import List



def bullet_list(items: List[str]) -> str:

    """Creates a bulleted list from a list of strings.



    Args:

        items: A list of strings to be formatted as a bulleted list.



    Returns:

        A bulleted list as a single string.

    """

    bulleted_list = ""

    for item in items:

        bulleted_list += f"- {item}\n"

    return bulleted_list

