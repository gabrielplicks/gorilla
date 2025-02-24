from typing import List



def get_longest_run(numbers: List[int]) -> int:

    """Calculates the length of the longest consecutive run of numbers in a list.



    Args:

        numbers: A list of numbers.



    Returns:

        The length of the longest consecutive run of numbers in the list.

    """

    longest_run_length = 0

    current_run_length = 0

    previous_number = None



    for number in numbers:

        if previous_number is None or number == previous_number + 1:

            current_run_length += 1

        else:

            current_run_length = 1

        longest_run_length = max(longest_run_length, current_run_length)

        previous_number = number



    return longest_run_length

