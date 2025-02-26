from typing import List



def group_zeros_and_ones(input_list: List[int]) -> List[int]:

    """Groups all the 0's together and all the 1's together in a list.



    Args:

        input_list: A list of 0's and 1's.



    Returns:

        A list of 0's and 1's with all the 0's grouped together and all the 1's grouped together.

    """

    zeros = []

    ones = []

    for num in input_list:

        if num == 0:

            zeros.append(num)

        else:

            ones.append(num)

    return zeros + ones

