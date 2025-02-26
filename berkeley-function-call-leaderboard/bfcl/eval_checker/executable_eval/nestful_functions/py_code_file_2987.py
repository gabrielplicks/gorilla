from typing import List, Union



def get_last_elements(lst: List[Union[List[int], List[str]]]) -> Union[str, str]:

    """Returns a string consisting of the last elements of each element in a list of lists.



    Args:

        lst: A list of lists.



    Returns:

        A string consisting of the last elements of each element in the list of lists.

        If any of the elements in the list are not lists themselves, the function returns an error message.

    """

    if not all(isinstance(item, list) for item in lst):

        return "Elements in the list must be lists themselves"

    return ''.join(map(str, map(lambda x: x[-1], lst)))

