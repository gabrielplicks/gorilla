import re



def find_sum_of_numbers_in_string(text: str) -> float:

    """Calculates the sum of all decimal numbers found in the input string.



    Args:

        text: The input string to search for decimal numbers.



    Returns:

        The sum of all decimal numbers found in the input string.

    """

    sum_of_numbers = 0.0

    numbers = re.findall(r'\d+\.?\d*', text)

    for number in numbers:

        sum_of_numbers += float(number)

    return sum_of_numbers

