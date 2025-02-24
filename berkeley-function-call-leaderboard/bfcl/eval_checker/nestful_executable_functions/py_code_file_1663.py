from typing import List



def chunk_the_list(l: List[int], n: int) -> List[List[int]]:

    """Divides a list into n equal-sized chunks.



    Args:

        l: The list to be divided.

        n: The number of chunks.



    Raises:

        ValueError: If n is not greater than 0.

    """

    if n <= 0:

        raise ValueError("n must be greater than 0")

    chunk_size = len(l) // n

    chunks = [l[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]

    if len(l) % n != 0:

        chunks[-1].extend(l[n * chunk_size:])



    return chunks

