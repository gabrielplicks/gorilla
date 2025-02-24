from typing import List, Tuple, Any



class TargetNotFound(Exception):

    """Custom exception raised when the target value is not found in the list."""

    def __init__(self, message: str):

        super().__init__(message)



def find_target(list_of_lists: List[List[Any]], target: Any) -> Tuple[int, int]:

    """Finds the indices of the target value in a list of lists.



    Args:

        list_of_lists: The list of lists to search.

        target: The target value to find.



    Raises:

        TargetNotFound: If the target value is not found in the list.

    """

    for i, inner_list in enumerate(list_of_lists):

        for j, value in enumerate(inner_list):

            if value == target:

                return (i, j)

    raise TargetNotFound(f'Target value {target} not found in the list.')

