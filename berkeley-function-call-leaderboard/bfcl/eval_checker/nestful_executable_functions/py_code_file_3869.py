from typing import List, Any



def convert_to_string_array(array: List[Any]) -> List[str]:

    """Converts an input array of any data type to a string array.



    Args:

        array: The input array to be converted.



    Returns:

        A string array where each element is the string representation of the corresponding element in the input array.

    """

    return list(map(str, array))

