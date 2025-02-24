import re

from typing import List



def parse_nmap_output(filename: str) -> List[str]:

    """Parses the output of an Nmap scan and returns a list of IP addresses.

    Args:

        filename: The name of the file containing the Nmap output.

    """

    regex = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"

    ip_addresses = []

    with open(filename, "r") as f:

        for line in f:

            match = re.search(regex, line)

            if match:

                ip_addresses.append(match.group())

    return ip_addresses

