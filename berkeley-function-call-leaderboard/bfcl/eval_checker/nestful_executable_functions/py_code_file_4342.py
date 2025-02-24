from typing import List



def string_list(items: List[str]) -> str:

    """Converts a list of strings to a single string, with each item separated by a comma and space.

    If there are more than two items in the list, an 'and' is inserted before the last item.

    Args:

        items: A list of strings.

    Returns:

        A single string with the items separated by commas and spaces, and an 'and' before the last item if there are more than two items.

    """

    if not items:

        return ""



    if len(items) == 1:

        return items[0]



    result = []

    for i, item in enumerate(items):

        if i == len(items) - 1:

            result.append(f"and {item}")

        else:

            result.append(f"{item}, ")



    return "".join(result)

