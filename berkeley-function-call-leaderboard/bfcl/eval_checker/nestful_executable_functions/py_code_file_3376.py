from typing import List, Tuple



def xs_and_ys(nums: List[int], xs: List[int], ys: List[int]) -> Tuple[List[int], List[int]]:

    """

    Separates even and odd numbers from a list of integers into two separate lists.

    Args:

        nums: A list of integers.

        xs: An empty list to store even numbers.

        ys: An empty list to store odd numbers.

    """

    for num in nums:

        if num % 2 == 0:

            xs.append(num)

        else:

            ys.append(num)

    return (xs, ys)

