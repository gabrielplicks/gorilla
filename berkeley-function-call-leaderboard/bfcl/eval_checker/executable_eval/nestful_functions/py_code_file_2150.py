from typing import List



def str_format_from_array(array: List[str]) -> str:

    """Creates a formatted string from an array of strings.



    Each element in the input array is concatenated with itself, separated by a colon `:`,

    and enclosed in curly brackets `{}`.



    Args:

        array: A list of strings.



    Returns:

        A formatted string.

    """

    output_str = ""



    for element in array:

        output_str += "{" + element + ":" + element + "}"

        if element != array[-1]:

            output_str += ":"



    return output_str

