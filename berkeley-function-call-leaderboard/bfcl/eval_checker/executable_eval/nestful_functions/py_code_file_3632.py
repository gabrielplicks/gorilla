from typing import List



def partition_elements(total_elements: int, num_partitions: int) -> List[int]:

    """Returns a list of integers representing the number of elements in each partition for a given total number of elements and the desired number of partitions.

    The number of elements in each partition should be as close as possible to the number of elements divided by the number of partitions, with the difference between them being at most 1.

    Args:

        total_elements: The total number of elements.

        num_partitions: The desired number of partitions.

    """

    partition_size = total_elements // num_partitions

    num_remaining_elements = total_elements % num_partitions

    partition_sizes = [partition_size] * num_partitions

    for i in range(num_remaining_elements):

        partition_sizes[i] += 1

    return partition_sizes

