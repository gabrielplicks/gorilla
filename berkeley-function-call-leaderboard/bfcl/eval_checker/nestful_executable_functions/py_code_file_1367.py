import re



def parse_even_sum(s: str) -> int:

    """Calculates the sum of all even numbers in a string containing multiple numbers separated by commas.



    Args:

        s: The input string containing multiple numbers separated by commas.



    Returns:

        The sum of all even numbers in the input string.

    """

    numbers = re.findall(r'\d+', s)

    even_numbers = [int(num) for num in numbers if int(num) % 2 == 0]

    return sum(even_numbers)

