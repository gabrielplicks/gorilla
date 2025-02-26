from typing import List



def merge_list(alist: List[int], blist: List[int]) -> List[int]:

    """Merges two sorted lists into one.



    Args:

        alist: The first sorted list.

        blist: The second sorted list.



    Returns:

        A new list that is the result of merging the two input lists.

    """

    i, j = 0, 0

    result = []



    while i < len(alist) and j < len(blist):

        if alist[i] < blist[j]:

            result.append(alist[i])

            i += 1

        else:

            result.append(blist[j])

            j += 1

    result += alist[i:]

    result += blist[j:]



    return result

