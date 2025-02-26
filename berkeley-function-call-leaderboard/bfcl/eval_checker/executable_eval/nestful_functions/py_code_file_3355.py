from typing import List



def find_intervals(numbers: List[int]) -> List[List[int]]:

    """Finds non-overlapping intervals that cover all the given integers.



    Args:

        numbers: A non-empty list of integers.



    Returns:

        A list of non-overlapping intervals that cover all the given integers.

    """

    numbers.sort()

    intervals = []

    current_interval = None

    for num in numbers:

        if current_interval is None or num == current_interval[1] + 1:

            if current_interval is None:

                current_interval = [num, num]

            else:

                current_interval[1] = num

        else:

            intervals.append(current_interval)

            current_interval = [num, num]

    if current_interval is not None:

        intervals.append(current_interval)



    return intervals

