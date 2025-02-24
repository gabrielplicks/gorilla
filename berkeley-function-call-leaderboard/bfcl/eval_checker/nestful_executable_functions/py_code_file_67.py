import socket



def send_data(host: str, port: int, data: str) -> None:

    """Sends data over a TCP socket to the given host and port.



    The data is sent as a byte array and is sent in the order it is received.

    The function raises an exception if the host or port cannot be reached or if

    any other I/O error occurs during sending.



    Args:

        host: The host to send the data to.

        port: The port to send the data to.

        data: The data to send.

    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)



    try:

        sock.connect((host, port))

        sock.sendall(bytearray(data))

    except Exception as e:

        raise e

    finally:

        sock.close()

