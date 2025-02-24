from typing import List, Dict

import sqlite3



def get_customers_with_orders() -> List[Dict[str, str]]:

    """Returns a list of customers who have placed orders in a database.



    Args:

        None



    Returns:

        A list of customers (represented as dictionaries with keys 'id' and 'name') who have placed at least one order.

    """

    conn = sqlite3.connect('database.db')

    cursor = conn.cursor()



    query = "SELECT * FROM customers INNER JOIN orders ON customers.id = orders.customer_id"

    cursor.execute(query)



    results = []

    for row in cursor.fetchall():

        customer = {

            'id': row[0],

            'name': row[1]

        }

        results.append(customer)



    cursor.close()

    conn.close()



    return results

