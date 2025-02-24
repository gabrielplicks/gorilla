import socket



def create_tcp_server(port: int, backlog: int = 10, reuse_addr: bool = True) -> socket.socket:

    """Creates a server socket and starts listening on a specified port.



    Args:

        port: the port number to listen on.

        backlog: the number of pending connections that can be queued (default: 10).

        reuse_addr: a boolean indicating whether to set the SO_REUSEADDR flag on the socket (default: True).



    Returns:

        The created server socket object.

    """

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, reuse_addr)

    server_socket.bind(('', port))

    server_socket.listen(backlog)



    return server_socket

