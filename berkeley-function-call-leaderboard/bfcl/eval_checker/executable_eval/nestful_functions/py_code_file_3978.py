from typing import Dict



def compute_total_value(tree: Dict[str, any]) -> float:

    """Computes the total value of a hierarchical data structure of products.



    Args:

        tree: A dictionary representing a product tree.



    Returns:

        The total value of the product tree.

    """

    if len(tree['children']) == 0:

        return tree['price']

    total_price = 0

    for child in tree['children']:

        total_price += compute_total_value(child)

    total_price += tree['price']

    return total_price

