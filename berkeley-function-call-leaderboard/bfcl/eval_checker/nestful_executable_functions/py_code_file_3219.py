from typing import Union



def copy_and_discount(amount: Union[int, float], discount: Union[int, float]) -> Union[int, float]:

    """Copies the amount to a new variable new_amount, then applies the discount percentage to new_amount.

    Returns the new_amount with the discount applied.

    Args:

        amount: The original amount.

        discount: The discount percentage.

    """

    new_amount = amount

    discounted_amount = new_amount * discount

    new_amount -= discounted_amount

    return new_amount

