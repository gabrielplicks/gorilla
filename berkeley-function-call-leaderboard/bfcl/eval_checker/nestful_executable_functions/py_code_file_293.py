from typing import List



def make_adder(numbers: List[int]) -> callable:

    """Creates a function that takes a number and returns the sum of all numbers in the list plus the input number.



    Args:

        numbers: A list of numbers.



    Returns:

        A function that takes a number and returns the sum of all numbers in the list plus the input number.

    """

    def adder(num: int) -> int:

        """Adds the input number to the sum of all numbers in the list.



        Args:

            num: The input number.



        Returns:

            The sum of all numbers in the list plus the input number.

        """

        return sum(numbers) + num

    return adder

