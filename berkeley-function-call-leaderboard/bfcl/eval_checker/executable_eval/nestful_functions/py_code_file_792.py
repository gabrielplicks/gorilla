from typing import List



def divide_list_into_dimensions(array: List[int], dimensions: int) -> List[List[int]]:

    """Divides an array into a specified number of dimensions, with each dimension containing as many elements as possible.

    The elements should be distributed in a round-robin fashion, i.e., the first dimension should contain the first few elements,

    then the second dimension, and so on. The function returns a list of dimensions, where each dimension is also a list.

    Args:

        array: The input array to be divided.

        dimensions: The number of dimensions to divide the array into.

    """

    result = [[] for _ in range(dimensions)]

    for i, element in enumerate(array):

        result[i % dimensions].append(element)

    return result

