from typing import List



def sum_and_index(lst: List[int]) -> List[str]:

    """Calculates the sum of each element and its index position in a given list.

    If the length of the given list is 4 or less, '!' is added to the end of each element.

    Args:

        lst: A list of integers.

    """

    new_lst = []



    for idx, element in enumerate(lst):

        new_element = element + idx



        if len(lst) <= 4:

            new_element += '!'



        new_lst.append(new_element)



    return new_lst

