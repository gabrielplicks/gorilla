import csv

from typing import Dict



def read_product_prices(filename: str) -> Dict[str, float]:

    """Reads a CSV file containing a table of products and their prices, and returns a dictionary that maps product names to their prices.



    Args:

        filename: The name of the CSV file.



    Returns:

        A dictionary containing the product names as keys and their prices as values.

    """

    with open(filename, 'r') as file:

        reader = csv.reader(file)

        next(reader)

        prices = {}

        for row in reader:

            product, price = row

            prices[product] = float(price)



        return prices

