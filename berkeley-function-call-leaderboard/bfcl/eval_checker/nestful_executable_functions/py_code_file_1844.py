import re

from typing import Union



def extract_last_error(log_file_path: str) -> Union[str, int]:

    """Reads a log file and extracts the last line that matches the regular expression pattern `^ERROR: [0-9]+$`.

    Returns the number after `ERROR: `. If there is no line that matches this pattern, returns `'No error'`.



    Args:

        log_file_path: The path to the log file.



    Returns:

        The last error number or `'No error'`.

    """

    with open(log_file_path, 'r') as f:

        lines = f.readlines()



    last_error_num = 'No error'

    for line in lines:

        match = re.search(r'^ERROR: ([0-9]+)$', line)

        if match:

            last_error_num = match.group(1)



    return last_error_num

