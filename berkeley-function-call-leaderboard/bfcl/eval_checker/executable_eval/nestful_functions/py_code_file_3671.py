from typing import List



def find_sum_of_integers(input_list: List[int]) -> int:

    """Calculates the sum of all integers in a list.



    Args:

        input_list: A list of integers.



    Raises:

        Exception: If the input list is empty.

        Exception: If any other error occurs.

    """

    try:

        if len(input_list) > 0:

            return sum(input_list)

        else:

            raise Exception("Input list is empty")

    except Exception as e:

        raise Exception("Invalid input: {}".format(e))

