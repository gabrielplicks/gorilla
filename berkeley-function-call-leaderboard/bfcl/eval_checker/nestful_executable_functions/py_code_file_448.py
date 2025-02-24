from typing import List



def move_negatives(lst: List[int]) -> List[int]:

    """Moves all negative numbers to the back of the list.



    Args:

        lst: A list of integers.



    Returns:

        A new list with all negative numbers at the back.

    """

    new_lst = lst[:]  # Making a shallow copy of the input list

    for i, num in enumerate(lst):

        if num < 0:

            new_lst.remove(num)

            new_lst.append(num)

    return new_lst

