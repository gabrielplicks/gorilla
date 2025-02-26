from typing import List



def average_consecutive(numbers: List[float]) -> List[float]:

    """Calculates the average of each number and the number before it in a list.



    Args:

        numbers: A list of numbers.



    Raises:

        TypeError: If the input is not a list or if any of the elements in the list are not numbers.

        ValueError: If the input list has fewer than two elements.



    Returns:

        A list of the averages of consecutive elements in the input list.

    """

    if not isinstance(numbers, list):

        raise TypeError('Input must be a list.')



    if len(numbers) < 2:

        raise ValueError('Input list must have at least two elements.')



    averages = []



    for i in range(1, len(numbers)):

        average = (numbers[i - 1] + numbers[i]) / 2

        averages.append(average)



    return averages

