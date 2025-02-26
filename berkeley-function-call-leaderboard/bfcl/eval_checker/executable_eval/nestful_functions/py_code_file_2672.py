from typing import List



def divide_chunks(lst: List[int], size: int) -> List[List[int]]:

    """Divides a list into fixed-size chunks, where the last chunk may be smaller than the predefined chunk size.



    Args:

        lst: The list to be divided into chunks.

        size: The desired chunk size.



    Returns:

        A list of sublists, where each sublist contains a fixed number of elements from `lst`.

    """

    chunks = []

    start_idx = 0

    end_idx = start_idx + size

    while end_idx <= len(lst):

        chunks.append(lst[start_idx:end_idx])

        start_idx = end_idx

        end_idx = start_idx + size

    if start_idx < len(lst):

        chunks.append(lst[start_idx:])

    return chunks

