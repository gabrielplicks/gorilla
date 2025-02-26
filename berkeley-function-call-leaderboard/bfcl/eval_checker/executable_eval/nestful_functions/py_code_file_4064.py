from typing import List



def get_all_subs(lst: List[int], acc: List[int] = None) -> List[List[int]]:

    """Generates all subsequences of a list of integers using tail recursion.



    Args:

        lst: The input list of integers.

        acc: The accumulator list used to store the subsequences.

            Defaults to an empty list if not provided.



    Returns:

        A list of all subsequences of the input list.

    """

    if acc is None:

        acc = []



    if not lst:

        return [acc]



    head = lst[0]



    return get_all_subs(lst[1:], acc) + get_all_subs(lst[1:], acc + [head])

