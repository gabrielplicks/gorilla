from typing import List



def create_filenames(nums: List[int], prefix: str) -> List[str]:

    """Creates a list of filenames using a pattern based on the values of nums and prefix.



    Args:

        nums: A list of integers.

        prefix: A string to be included in the filenames.



    Returns:

        A list of filenames.

    """

    filenames = []

    for num in nums:

        filename = prefix + str(num) + ".txt"

        filenames.append(filename)

    return filenames

