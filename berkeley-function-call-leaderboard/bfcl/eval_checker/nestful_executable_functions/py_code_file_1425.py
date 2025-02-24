from typing import Union



def calculate_profit(selling_price: Union[int, float], buying_price: Union[int, float]) -> float:

    """Calculates the profit from a financial trading-related scenario.



    Args:

        selling_price: The price at which the asset is sold.

        buying_price: The price at which the asset is bought.

    """

    profit = selling_price - buying_price

    return profit

