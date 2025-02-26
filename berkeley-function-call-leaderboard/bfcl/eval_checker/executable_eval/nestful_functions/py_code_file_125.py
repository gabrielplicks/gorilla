import socket

import time



def isConnected(address: tuple) -> bool:

    """Establishes a connection to the given TCP socket address and returns a boolean value indicating whether the connection is successful.



    Args:

        address: A tuple containing the host name and port number of the TCP socket address.



    Returns:

        True if the connection is successful, False otherwise.

    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)



    try:

        start = time.time()

        sock.connect(address)

        return True

    except Exception as e:

        elapsed = time.time() - start

        print(f"Error connecting to {address}: {e.strerror} (elapsed time: {elapsed} seconds)")

        return False

    finally:

        sock.close()

