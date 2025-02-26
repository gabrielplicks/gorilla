from typing import List



def retrieve_cyclic_elements(ls: List[int], limit: int) -> List[int]:

    """Retrieves elements from a list in a cyclic order.



    Args:

        ls: The list of numbers.

        limit: The number of elements to retrieve.



    Returns:

        A list of elements retrieved from `ls` in a cyclic order.

    """

    if not ls:

        return []



    cyclic_elements = []

    index = 0



    while len(cyclic_elements) < limit:

        cyclic_elements.append(ls[index])

        index = (index + 1) % len(ls)



    return cyclic_elements

