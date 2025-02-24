from typing import List



def product_without_division(numbers: List[int]) -> List[int]:

    """Calculates the product of all other elements except the current one for each element in a list.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of the same length where each element is the product of all other elements except the element at the current index.

    """

    output = []

    for i in range(len(numbers)):

        product = 1

        for j in range(len(numbers)):

            if i != j:

                product *= numbers[j]

        output.append(product)

    return output

