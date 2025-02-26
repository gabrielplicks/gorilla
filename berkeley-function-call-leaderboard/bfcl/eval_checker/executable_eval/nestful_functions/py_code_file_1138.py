from datetime import datetime

from collections import defaultdict

from typing import Dict, Any



def stock_prices_per_day(stock_prices: Dict[str, Any]) -> Dict[str, list]:

    """Returns a dictionary of stock prices per day.



    Args:

        stock_prices: A dictionary of stock prices at different timestamps.



    Returns:

        A dictionary of stock prices per day. If the prices for a given day are in a single timestamp,

        the value is that price. If there are multiple prices for a day, the value is an array of prices.

    """

    prices_per_day = defaultdict(list)

    for timestamp, price in stock_prices.items():

        day = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d')

        prices_per_day[day].append(price)

    return prices_per_day

