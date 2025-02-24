from typing import List, Union



def first_element_or_none(first_list: List[Union[int, str]], second_list: List[Union[int, str]]) -> Union[int, str, None]:

    """Returns the first element of the first list, or None if the first list is empty.



    Args:

        first_list: The first list.

        second_list: The second list.

    """

    if not first_list:

        return None

    return first_list[0]

