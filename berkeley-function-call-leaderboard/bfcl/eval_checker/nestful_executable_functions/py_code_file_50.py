from typing import List



def compute_returns(prices: List[float]) -> List[float]:

    """Computes the daily return (the percent change in price relative to the previous day's closing price) using list comprehension and the `zip` operator, without using a for loop.



    Args:

        prices: A list of prices that represent a stock's daily closing price over a period of time.



    Returns:

        A list of the same length that contains the daily return (the percent change in price relative to the previous day's closing price).

    """

    return [(today_price - prev_price) / prev_price for today_price, prev_price in zip(prices[1:], prices[:-1])]

