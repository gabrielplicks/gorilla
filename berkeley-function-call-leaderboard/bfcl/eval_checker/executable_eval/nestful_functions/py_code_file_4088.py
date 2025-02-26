from typing import List, Dict



def sum_by_index(collection: List[Dict], index: str) -> List[int]:

    """Calculates the sum of the values of a given index in a collection of dictionaries.



    Args:

        collection: A list of dictionaries.

        index: A key in the dictionaries.



    Returns:

        A list of integers where each integer is the sum of the corresponding index values in each dictionary.

        If the index key is not found in any of the dictionaries, an empty list is returned.

    """

    sum_list = []

    flag = False



    for dictionary in collection:

        value = dictionary.get(index)



        if isinstance(value, int):

            sum_list.append(value)

            flag = True



    if not flag:

        sum_list = []



    return sum_list

