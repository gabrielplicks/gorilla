from typing import Union



def is_valid_xlsx_or_csv(file_path: str) -> bool:

    """Checks if a file is a valid .xlsx or .csv file.



    Args:

        file_path: The path of the file to check.



    Returns:

        True if the file is a valid .xlsx or .csv file, False otherwise.

    """

    file_parts = file_path.split('.')



    if len(file_parts) > 1:

        file_extension = file_parts[-1]



        if file_extension == 'xlsx' or file_extension == 'csv':

            return True



    return False

