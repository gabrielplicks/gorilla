from typing import List, Dict



def find_common_attributes(l1: List[Dict[str, str]], l2: List[Dict[str, str]]) -> List[str]:

    """Finds the common attribute names in two lists of objects.



    Args:

        l1: A list of objects, each with a 'name' attribute.

        l2: A list of objects, each with a 'name' attribute.



    Returns:

        A list of the common attribute names.

    """

    return list(set([obj['name'] for obj in l1]) & set([obj['name'] for obj in l2]))

