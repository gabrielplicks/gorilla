import socket



def get_free_port() -> int:

    """Gets a free port on the local machine using TCP socket programming.



    Args:

        None



    Returns:

        The available TCP port number.

    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.bind(('localhost', 0))

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    sock.bind(('localhost', 0))

    port = sock.getsockname()[1]

    sock.close()

    return port

