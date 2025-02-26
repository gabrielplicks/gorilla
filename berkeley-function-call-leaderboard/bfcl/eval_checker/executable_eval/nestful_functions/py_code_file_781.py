import csv

from typing import List, Tuple



class Product:

    def __init__(self, id: int, name: str, *args):

        self.id = id

        self.name = name



def read_products(csv_file: str) -> List[Product]:

    """Reads a CSV file containing a list of products and returns a list of product objects with only the `id` and `name` attributes.



    Args:

        csv_file: The path to the CSV file.



    Returns:

        A list of `Product` objects with only the `id` and `name` attributes.

    """

    with open(csv_file, 'r') as file:

        reader = csv.reader(file)

        rows = list(reader)

        product_list = [Product(*row[:2]) for row in rows]

        return product_list

