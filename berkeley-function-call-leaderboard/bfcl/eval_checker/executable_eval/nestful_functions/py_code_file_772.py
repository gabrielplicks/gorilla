import socket



def check_open_ports(hostnames: list[str], ports: list[int]) -> list[str]:

    """

    Checks if the given ports on the given hostnames are open.

    Returns a list of hostnames where the ports are open.

    Args:

        hostnames: A list of hostnames to check.

        ports: A list of ports to check.

    """

    open_hostnames = []

    for hostname in hostnames:

        for port in ports:

            try:

                socket.create_connection((hostname, port), timeout=1)

                open_hostnames.append(hostname)

                break

            except ConnectionRefusedError:

                continue



    return open_hostnames

