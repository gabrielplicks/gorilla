import re

from typing import Union



def parse_ping_result(output: str) -> Union[int, None]:

    """Parses the output of a ping command and returns the average latency in milliseconds.



    Args:

        output: The output string of a ping command.



    Returns:

        The average latency in milliseconds as an integer, rounded to the nearest integer.

        If the input string does not match the expected format, the function returns None.

    """

    match = re.search(r"rtt min/avg/max/mdev = \d+\.\d+/\d+\.\d+/\d+\.\d+/\d+\.\d+ ms", output)

    if match:

        latency_str = match.group().split("=")[1].strip().split("/")[1]

        average_latency = float(latency_str.split(".")[0])

        return int(round(average_latency))

    else:

        return None

