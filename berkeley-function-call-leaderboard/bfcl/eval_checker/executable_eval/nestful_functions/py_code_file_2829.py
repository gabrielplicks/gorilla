from typing import List



import heapq



def largest_element_in_heap(heap: List[int]) -> int:

    """Returns the largest element in a heap.



    Args:

        heap: A list representing a heap.



    Returns:

        The largest element in the heap.

    """

    return heapq.nlargest(1, heap)[0]

