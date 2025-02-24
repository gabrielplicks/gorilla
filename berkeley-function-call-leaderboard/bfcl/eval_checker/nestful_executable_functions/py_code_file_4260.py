import subprocess

from typing import Any



def check_command(command: str) -> bool:

    """Checks for the existence of a command on the system.



    Args:

        command: The command to check for.



    Returns:

        True if the command exists, False otherwise.

    """

    try:

        result = subprocess.run(["which", command], capture_output=True, text=True)

        return result.returncode == 0

    except Exception as e:

        print(f"An error occurred: {e}")

        return False

