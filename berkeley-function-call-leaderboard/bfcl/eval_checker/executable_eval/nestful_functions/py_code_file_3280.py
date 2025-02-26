from typing import List



def create_list_with_default_value(size: int, default_value: any) -> List[any]:

    """Creates an empty list with a specified size and initializes each element to a given value.



    Args:

        size: The size of the list.

        default_value: The default value for each element.



    Returns:

        A list of the specified size with each element initialized to the default value.

    """

    return [default_value for _ in range(size)]

