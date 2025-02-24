from typing import List



def generate_array_type(arr: List[str]) -> str:

    """Generates an array type string based on the provided array.



    Args:

        arr: A list of strings representing the elements of the array.



    Returns:

        A string representing the array type.

    """

    type_elements = []

    for element in arr:

        type_elements.append(str(element))

    return "[" + ', '.join(type_elements).strip() + "]"

