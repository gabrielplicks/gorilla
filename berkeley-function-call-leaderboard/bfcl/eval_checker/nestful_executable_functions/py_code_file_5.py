from typing import Dict, List



def map_nums_to_chars(nums: List[int]) -> Dict[str, int]:

    """Maps each integer in a list to its corresponding character in the ASCII table.



    Args:

        nums: A list of integers.



    Returns:

        A dictionary where each key is a character and each value is its corresponding integer.

    """

    char_set = {chr(num) for num in nums}

    return {char: ord(char) for char in char_set}

