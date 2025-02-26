import csv



def write_nested_list_to_csv(nested_list: list[list[int]], output_path: str) -> bool:

    """Writes a nested list of integers to a CSV file at the specified output path.

    Args:

        nested_list: The nested list of integers to write to the CSV file.

        output_path: The path to the output CSV file.

    Returns:

        True if the file was written successfully, False otherwise.

    """

    try:

        with open(output_path, 'w', newline='') as csv_file:

            writer = csv.writer(csv_file)

            for sublist in nested_list:

                writer.writerow(sublist)

        return True

    except Exception as e:

        print(f'Error: {e}')

        return False

