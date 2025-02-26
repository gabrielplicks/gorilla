import csv



def filter_csv_rows() -> None:

    """

    Reads a CSV file and filters out rows with a value greater than 150.

    Writes the remaining rows to a new CSV file.

    """

    with open('data.csv', 'r') as csv_file:

        reader = csv.DictReader(csv_file, delimiter=',')

        with open('data_filtered.csv', 'w') as csv_file_filtered:

            writer = csv.DictWriter(csv_file_filtered, fieldnames=reader.fieldnames, delimiter=',')

            writer.writeheader()

            for row in reader:

                if row['value'] and int(row['value']) <= 150:

                    writer.writerow(row)

