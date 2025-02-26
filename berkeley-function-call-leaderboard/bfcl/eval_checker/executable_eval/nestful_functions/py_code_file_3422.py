from typing import List



def filter_elements(lst: List[str], condition: List[str]) -> List[str]:

    """Filters elements in a given list based on a specified condition.



    Args:

        lst: The list to filter.

        condition: A list of strings specifying the condition for filtering.



    Returns:

        A new list containing only the elements that do not appear in the condition list.

    """

    return [element for element in lst if element not in condition]

