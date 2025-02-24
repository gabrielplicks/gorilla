import csv



def find_rows_with_value(file_path: str, value: str) -> list:

    """Finds rows in a CSV file where the second column contains a certain value.



    Args:

        file_path: The path to the CSV file.

        value: The value to search for in the second column.



    Returns:

        A list of rows where the second column contains the value.

    """

    rows = []

    with open(file_path, 'r') as csv_file:

        csv_reader = csv.reader(csv_file)

        for row in csv_reader:

            if row[1] == value:  # Check if the second column contains the desired value

                rows.append(row)

    return rows

