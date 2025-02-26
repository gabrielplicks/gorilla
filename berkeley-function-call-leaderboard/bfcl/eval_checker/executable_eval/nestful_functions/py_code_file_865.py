import re

from typing import List



def get_log_messages(filename: str) -> List[str]:

    """Extracts log messages from a file with a given format.



    Args:

        filename: The name of the log file.



    Returns:

        A list of log messages.

    """

    with open(filename, 'r') as f:

        content = f.read()

    pattern = r'\[(INFO|WARNING|ERROR)\]\s+(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{6}):\s*(.+)'

    matches = re.findall(pattern, content)

    return [m[2] for m in matches]

