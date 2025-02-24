import socket



def open_connection(host: str, port: int) -> socket.socket:

    """Creates a socket connection to a given host and port.



    Args:

        host: The host address to connect to.

        port: The port number to connect to.



    Returns:

        A socket object if the connection is successful, otherwise None.

    """

    try:

        sock = socket.socket()

        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        sock.connect((host, port))

        return sock

    except socket.error as e:

        print("Error connecting to host:", e)

