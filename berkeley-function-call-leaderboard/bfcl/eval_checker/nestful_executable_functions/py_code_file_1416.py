from typing import List, Union



def is_valid_element(element: Union[str, List]) -> bool:

    """Validates if a given element is a valid string or list.



    Args:

        element: The element to validate.



    Returns:

        True if the element is a valid string or list, False otherwise.

    """

    if isinstance(element, str):

        return len(element) > 0

    if isinstance(element, list) and element:

        return all(is_valid_element(child) for child in element)

    return False



def validate_list(lst: List[Union[str, List]]) -> bool:

    """Validates if a given list is a valid list of lists.



    Args:

        lst: The list to validate.



    Returns:

        True if the list is a valid list of lists, False otherwise.

    """

    if not isinstance(lst, list):

        return False

    return all(is_valid_element(element) for element in lst)

