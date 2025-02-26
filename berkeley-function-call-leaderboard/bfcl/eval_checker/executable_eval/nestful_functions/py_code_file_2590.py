from typing import List, Set



def get_unique_tag_names(tags: List[object]) -> List[str]:

    """Returns a list of unique tag names sorted in alphabetical order.



    Args:

        tags: A list of tags, each of which has a `.name` attribute.



    Returns:

        A list of unique tag names sorted in alphabetical order.

    """

    unique_names: Set[str] = set()

    for tag in tags:

        unique_names.add(tag.name)

    return sorted(unique_names)

