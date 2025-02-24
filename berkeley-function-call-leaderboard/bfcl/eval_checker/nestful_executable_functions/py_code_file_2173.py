import socket



def port_binding_race(port_number: int) -> socket.socket:

    """Binds to a specified port number using a socket, checking for race conditions.



    Args:

        port_number: The port number to bind.



    Returns:

        The socket object if the binding operation is successful.



    Raises:

        socket.error: If the binding operation fails.

    """

    try:

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.bind(('localhost', port_number))

    except OSError as e:

        if e.errno == socket.errorcode['EADDRINUSE']:

            new_port = port_number + 1

            return port_binding_race(new_port)

        else:

            raise e

    return sock

