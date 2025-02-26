from typing import List, Dict



def purchase_order_details(purchases: List[Dict[str, int or str or float]]) -> List[str]:

    """

    Returns a list of strings representing the purchase order details.

    Args:

        purchases: A list of dictionaries where each dictionary represents a purchase order object.

    """

    orders = []

    for purchase in purchases:

        order_id = purchase['order_id']

        supplier_name = purchase['supplier_name']

        quantity = purchase['quantity']

        amount_paid = purchase['amount_paid']



        order = 'Purchase Order {}: Bought {} items from {} for ${}'.format(

            order_id,

            quantity,

            supplier_name,

            amount_paid

        )

        orders.append(order)



    return orders

