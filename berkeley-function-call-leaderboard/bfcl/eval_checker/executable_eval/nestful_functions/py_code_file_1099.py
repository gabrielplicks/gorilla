from typing import List



class Element:

    def __init__(self, priority: int):

        self.priority = priority



def replace_with_priority(priority_list: List[Element]) -> List[int]:

    """Creates a copy of the priority_list with elements sorted in descending order based on their priority attribute,

    and then replaces each element with its priority attribute.



    Args:

        priority_list: A list of elements with a .priority attribute.



    Returns:

        A list of priorities in descending order.

    """

    priority_list_copy = sorted(priority_list, key=lambda x: x.priority, reverse=True)

    return [element.priority for element in priority_list_copy]



priority_list = [Element(5), Element(4), Element(3), Element(2), Element(1)]



priority_list_copy = replace_with_priority(priority_list)



print(priority_list_copy)

