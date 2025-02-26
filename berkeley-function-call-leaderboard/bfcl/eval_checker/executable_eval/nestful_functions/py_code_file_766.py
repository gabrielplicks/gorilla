from typing import Any



def is_homogeneous(data: Any) -> bool:

    """Checks if a data structure is homogeneous, meaning it contains only a single type of value.



    Args:

        data: The data structure to check.



    Returns:

        True if the data structure is homogeneous, False otherwise.

    """

    if isinstance(data, list):

        element_type = type(data[0])

        return all(isinstance(element, element_type) for element in data)

    else:

        return isinstance(data, type(data))

