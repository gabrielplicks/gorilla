from typing import List



def search_name(names: List[str], name_to_search: str) -> bool:

    """Searches for a given name in a list of names.

    Args:

        names: A list of names.

        name_to_search: The name to search for.

    """

    if len(names) == 0 or name_to_search == "":

        return False



    for name in names:

        if name == name_to_search:

            return True

    return False

