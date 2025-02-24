import csv

from typing import List, Dict



def write_csv_from_dict_list(csv_path: str, data: List[Dict[str, str]]) -> int:

    """Writes the contents of a list of dictionaries to a csv file.



    Args:

        csv_path: The path of the csv file to write.

        data: A list of dictionaries representing the data to write.



    Returns:

        The number of records written to the csv file successfully.

    """

    csv.register_dialect('custom_dialect', delimiter=',', quotechar='"')



    with open(csv_path, 'w', newline='') as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=data[0].keys(), dialect='custom_dialect')

        writer.writeheader()

        for row in data:

            writer.writerow(row)

    return len(data)

