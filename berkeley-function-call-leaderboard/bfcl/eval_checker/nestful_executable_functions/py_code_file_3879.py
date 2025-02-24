import csv

from typing import List, Tuple



def sort_csv_by_stock_value(filename: str) -> List[Tuple[str, float]]:

    """Loads data from a CSV file, parses it, and sorts it by the "stock_value" column in descending order.

    Returns a list of tuples, where each tuple contains the "id" and "stock_value" columns of the sorted CSV rows.



    Args:

        filename: The name of the CSV file.



    Returns:

        A list of tuples containing the "id" and "stock_value" columns of the sorted CSV rows.

    """

    with open(filename, "r") as csvfile:

        reader = csv.DictReader(csvfile)

        rows = []

        for row in reader:

            stock_value = float(row["stock_value"])

            rows.append((row["id"], stock_value))

        rows.sort(key=lambda x: x[1], reverse=True)

        return rows

