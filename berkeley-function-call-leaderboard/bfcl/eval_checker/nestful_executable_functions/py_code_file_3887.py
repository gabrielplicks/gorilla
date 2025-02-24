from typing import List



def cumulative_list(input_list: List[float]) -> List[List[float]]:

    """Creates a list of lists of the same values as the input list.

    Each inner list contains the original values up to and including the current value.

    Args:

        input_list: A list of numeric values.

    """

    output = []

    for i in range(len(input_list)):

        output.append(input_list[:i + 1])

    return output

