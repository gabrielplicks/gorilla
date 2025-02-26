from typing import List



def product_except_index(array: List[int]) -> List[int]:

    """Calculates the product of all elements in the array except for the current index.



    Args:

        array: A list of integers.



    Returns:

        A new list where the i-th element is equal to the product of all array elements except for the i-th element.

    """

    new_array = []

    for i in range(len(array)):

        product = 1

        for j in range(len(array)):

            if j != i:

                product *= array[j]

        new_array.append(product)

    return new_array

