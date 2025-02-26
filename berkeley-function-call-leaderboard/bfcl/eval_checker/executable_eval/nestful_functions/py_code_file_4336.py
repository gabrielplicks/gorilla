from typing import Dict

import csv



def create_dictionary_from_csv(csv_file: str) -> Dict[str, int]:

    """Creates a dictionary from a CSV file, keeping track of the number of rows and skipping the remaining rows if there are more than 10 rows.



    Args:

        csv_file: The path to the CSV file.



    Returns:

        A dictionary where each word is a key and its frequency is the corresponding value.

    """

    with open(csv_file, 'r') as f:

        reader = csv.reader(f)

        num_rows = 0

        dictionary = {}



        for row in reader:

            if num_rows == 10:

                break



            word, frequency = row

            dictionary[word] = int(frequency)

            num_rows += 1



        return dictionary

