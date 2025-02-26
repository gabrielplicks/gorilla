from typing import List



def batch_numbers(numbers: List[int]) -> List[List[int]]:

    """

    Takes a list of numbers as input and returns a list of lists of numbers.

    The returned list is the input list, but with each sublist containing the batch of numbers of the same length.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of lists of numbers.

    """

    result = [[]]

    for num in numbers:

        if len(result[-1]) == 0:

            result[-1].append(num)

        elif result[-1][-1] == num:

            result[-1].append(num)

        else:

            result.append([num])



    return result

