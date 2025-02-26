import subprocess

from typing import Union



def which(name: str) -> Union[str, None]:

    """Searches for an executable binary with a given name using the `which` command.

    Args:

        name: The name of the binary to search for.

    Returns:

        The absolute path of the binary if found, and None otherwise.

    """

    try:

        output = subprocess.check_output(["which", name])

        return output.decode("utf-8").strip()

    except subprocess.CalledProcessError:

        return None

    except Exception as e:

        print(f"Error: {e}")

        return None

