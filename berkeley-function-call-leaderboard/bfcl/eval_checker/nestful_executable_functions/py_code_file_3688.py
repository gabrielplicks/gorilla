def discount(price: float, discount_rate: int, duration: int) -> float:

    """Calculates the discounted price of an item given the original price, discount rate, and the number of years the discount is valid.



    Args:

        price: The original price of the item.

        discount_rate: The percentage of the original price to be discounted.

        duration: The number of years the discount is valid.



    Returns:

        The discounted price as a float.

    """

    if discount_rate == 0:

        return price

    else:

        discounted_price = price * (1 - discount_rate / 100) ** duration

        return discounted_price

