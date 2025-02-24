from typing import Dict



def count_errors(log_file: str) -> Dict[str, int]:

    """Counts the number of times each error message occurs in a log file.



    Args:

        log_file: The path to the log file.



    Returns:

        A dictionary with keys representing error messages and values representing

        the number of times that message occurred in the log file.

    """

    error_counts: Dict[str, int] = {}

    with open(log_file, 'r') as f:

        for line in f:

            error_message = line.split("Error")[1].strip()

            if error_message in error_counts:

                error_counts[error_message] += 1

            else:

                error_counts[error_message] = 1

    return error_counts

