import re

from typing import Dict



def extract_log_record(record: str) -> Dict[str, str]:

    """Extracts several pieces of information from a string that contains a log record.

    Args:

        record: The log record string.

    Returns:

        A dictionary containing the following keys: 'name', 'timestamp', and 'message'.

    """

    result: Dict[str, str] = {}

    m = re.match(r'^([^ ]*)', record)

    if m:

        result['name'] = m.group(1)

    m = re.search(r'\[([^]]*)\]', record)

    if m:

        result['timestamp'] = m.group(1)

    m = re.search(r'\](.*)', record)

    if m:

        result['message'] = m.group(1).strip()



    return result

