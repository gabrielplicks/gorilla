from typing import List, Dict



def get_visited_ancestors(node: Dict) -> List[Dict]:

    """Returns a list of all ancestor dictionaries of the dictionaries with a `visited` key set to `True`.



    Args:

        node: A dictionary containing a key called `children` that holds a list of child dictionaries.



    Returns:

        A list of ancestor dictionaries.

    """

    ancestors = []

    if node.get('visited'):

        ancestors.append(node)

    for child in node.get('children', []):

        ancestors.extend(get_visited_ancestors(child))

    return ancestors

