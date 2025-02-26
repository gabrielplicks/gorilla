from typing import List



def product_modulo(nums: List[int], modulo: int) -> int:

    """Calculates the product of a list of integers modulo a given number.

    Args:

        nums: A list of integers.

        modulo: A large prime number.

    """

    product = 1

    for num in nums:

        product = (product * num) % modulo

    return product

