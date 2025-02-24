from typing import List



def even_odd_numbers(num_arr: List[int]) -> dict:

    """Creates a dictionary containing two arrays: one with the even numbers and the other with the odd numbers.

    If the input array contains no odd numbers, sets the key of the dictionary to 'even' and returns an empty array for the 'odd' key.

    Args:

        num_arr: A list of numbers.

    """

    even_nums = [num for num in num_arr if num % 2 == 0]

    odd_nums = [num for num in num_arr if num % 2 == 1]

    if odd_nums:

        return {'even': even_nums, 'odd': odd_nums}

    else:

        return {'even': even_nums, 'odd': []}

