from typing import List, Union



def intersect_lists(list1: Union[List[int], None], list2: Union[List[int], None]) -> List[int]:

    """Calculates the intersection of two lists.

    Args:

        list1: A list of integers or None.

        list2: A list of integers or None.

    """

    try:

        intersection = []

        if list1 is None or list2 is None:

            return intersection



        for elem in list1:

            if elem in list2:

                intersection.append(elem)



        return intersection

    except:

        return []

