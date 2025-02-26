from typing import List, Generator



def convert_to_strings_list_comprehension(numbers: List[float]) -> List[str]:

    """Converts all the elements in a list of numbers to strings using a list comprehension.



    Args:

        numbers: The list of numbers to convert to strings.



    Returns:

        A list of strings representing the input numbers.

    """

    return [str(x) for x in numbers]



def convert_to_strings_generator(numbers: List[float]) -> Generator[str, None, None]:

    """Converts all the elements in a list of numbers to strings using a generator expression.



    Args:

        numbers: The list of numbers to convert to strings.



    Returns:

        A generator that lazily generates the string representation of each number.

    """

    return (str(x) for x in numbers)

