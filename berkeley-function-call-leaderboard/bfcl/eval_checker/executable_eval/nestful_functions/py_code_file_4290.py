from typing import List



def count_matches(nums1: List[int], nums2: List[int]) -> int:

    """Calculates the number of common elements between two collections.



    Args:

        nums1: The first collection of numbers.

        nums2: The second collection of numbers.



    Returns:

        The number of common elements between the two collections.

    """

    nums1_set = set(nums1)

    nums2_set = set(nums2)



    intersection = nums1_set.intersection(nums2_set)



    return len(intersection)

