import datetime

from typing import Tuple



def process_log_line(log_line: str) -> Tuple[datetime.datetime, str]:

    """Processes a log line string and extracts the timestamp and message.



    Args:

        log_line: The log line string to process.



    Returns:

        A tuple containing the timestamp and message.

    """

    timestamp_str, message = log_line.split('Message: ', 1)

    timestamp_str = timestamp_str.strip('Time: ')

    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

    message = repr(message)



    return (timestamp, message)

