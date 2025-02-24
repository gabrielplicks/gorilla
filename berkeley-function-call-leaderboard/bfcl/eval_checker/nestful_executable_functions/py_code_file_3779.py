from typing import List



def read_log_file_and_get_ips(file_path: str) -> List[str]:

    """Reads a log file and returns a list of unique IP addresses.



    Args:

        file_path: The path to the log file.



    Raises:

        Exception: If the file path is not a .log file.

        Exception: If there is an error reading the log file.

    """

    if not file_path.endswith('.log'):

        raise Exception('Invalid file path: File must be a .log file.')

    try:

        with open(file_path, 'r') as f:

            ips = set()

            for line in f:

                ip = line.split()[0]

                ips.add(ip)

    except (FileNotFoundError, PermissionError) as e:

        raise Exception(f'Error reading log file: {e}')

    return list(ips)

