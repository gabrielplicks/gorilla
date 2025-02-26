from typing import List



def set_diff(A: List[str], B: List[str]) -> List[str]:

    """Returns a list of strings in A but not in B.



    The order of the returned strings should be the same as that of A.



    Args:

        A: A list of strings.

        B: A list of strings.



    """

    diff_list = []



    for string in A:

        if string not in B:

            diff_list.append(string)



    return diff_list

