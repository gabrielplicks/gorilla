import logging

from typing import List



def process_integers(numbers: List[str]) -> List[int]:

    """Processes a list of integers and returns a list of unique values in a set.

    If there is an error during processing (e.g., a non-integer value is encountered),

    logs the error and continues processing the rest of the list.



    Args:

        numbers: A list of integers as strings.



    Returns:

        A list of unique integers.

    """

    unique_numbers = set()



    for num in numbers:

        try:

            num = int(num)

            unique_numbers.add(num)

        except ValueError as err:

            logging.error(f"Error processing number: {err}")



    return list(unique_numbers)

