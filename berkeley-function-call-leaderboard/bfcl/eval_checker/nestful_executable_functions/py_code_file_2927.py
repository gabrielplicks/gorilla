from typing import Union



def calculate_profit_percentage(initial_price: Union[int, float], final_price: Union[int, float]) -> float:

    """Calculates the percentage change in the value of a product between its initial price and the final price.



    Args:

        initial_price: The initial price of the product.

        final_price: The final price of the product.



    Returns:

        The percentage change as a float value rounded to 2 decimal places.

    """

    difference = final_price - initial_price

    fractional_change = difference / initial_price

    percentage_change = fractional_change * 100

    rounded_percentage_change = round(percentage_change, 2)



    return rounded_percentage_change

