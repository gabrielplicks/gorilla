from typing import Dict



def replace_cheapest_item(inventory: Dict[str, float]) -> Dict[str, float]:

    """Sorts the items in the inventory by their price in ascending order and replaces the price of the cheapest item with a fixed price of 0.



    Args:

        inventory: A dictionary where the keys are the names of the items and the values are their prices.



    Returns:

        A dictionary where the keys are the names of the items and the values are their prices, with the price of the cheapest item replaced with 0.

    """

    sorted_items = sorted(inventory.items(), key=lambda x: x[1])

    cheapest_price = sorted_items[0][1]



    sorted_items[0] = (sorted_items[0][0], 0)

    return dict(sorted_items)

