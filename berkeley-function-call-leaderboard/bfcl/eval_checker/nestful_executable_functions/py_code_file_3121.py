import csv



def get_countries_from_csv(csv_file: str) -> list[str]:

    """Extracts the country name for each row in a CSV file and returns a list of the extracted country names as strings.



    Args:

        csv_file: The path to the CSV file.



    Returns:

        A list of the extracted country names as strings.

    """

    with open(csv_file, 'r') as f:

        reader = csv.DictReader(f)

        return [row['country'] for row in reader]

