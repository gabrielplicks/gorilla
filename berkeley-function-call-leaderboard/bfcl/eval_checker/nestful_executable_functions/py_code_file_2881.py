import re



def get_host_and_port(input_string: str) -> dict:

    """Extracts the host and port from an input string in the format "host:port".



    Args:

        input_string: The input string containing the host and port.



    Returns:

        A dictionary containing the values for `host` and `port`. If the input

        string does not have a port, the resulting dictionary will only contain

        the `host` key.

    """

    match = re.search(r"^([a-zA-Z0-9.-]+):(\d+)$", input_string)

    if match:

        host = match.group(1)

        port = int(match.group(2))

        return {"host": host, "port": port}

    else:

        return {"host": input_string}

