from collections import Counter

from typing import List, Dict



def generate_frequency_table(nums: List[float]) -> Dict[float, int]:

    """

    Generates a frequency table from a list of numbers.



    Args:

        nums: A list of numbers.



    Returns:

        A dictionary that maps each unique number to its frequency.

    """

    try:

        if not isinstance(nums, list) or not nums:

            return {}

        freq_table = Counter(nums)

        return freq_table

    except Exception as e:

        print(f"An error occurred: {repr(e)}")

        return {}

