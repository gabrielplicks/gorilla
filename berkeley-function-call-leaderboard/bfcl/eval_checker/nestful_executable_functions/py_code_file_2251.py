from typing import List, Tuple



def convert_x_to_list(input_list: List[Tuple[int, int]]) -> List[Tuple[List[int], int]]:

    """Converts the x component of each tuple in the input list to a list.



    Args:

        input_list: A list of tuples of the form (x, y).



    Returns:

        A list of tuples with the x component converted to a list.

    """

    result = []

    for x, y in input_list:

        result.append([x], y)

    return result

