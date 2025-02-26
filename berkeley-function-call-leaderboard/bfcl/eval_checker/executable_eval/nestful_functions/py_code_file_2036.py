from typing import Tuple



def buy_cookies(buy_n: int, get_m_for_free: int) -> float:

    """Calculates the total cost in dollars of cookies purchased.

    Args:

        buy_n: The number of cookies to purchase.

        get_m_for_free: The number of cookies received for free when buying a certain number of cookies.

    """

    free_cookies = buy_n // (get_m_for_free + 1) * get_m_for_free

    remaining_cookies = buy_n - free_cookies

    cost = remaining_cookies * 1.5



    return cost

