import heapq



def find_nth_smallest(nums: list[int], n: int) -> int:

    """Efficiently finds the nth smallest element in a list of unsorted integers.



    Args:

        nums: A list of unsorted integers.

        n: The position of the smallest element to find.



    Returns:

        The nth smallest element in the list.

    """

    heap = []

    for num in nums:

        if len(heap) < n:

            heapq.heappush(heap, num)

        elif num < heap[0]:

            heapq.heapreplace(heap, num)

    return heap[0]

