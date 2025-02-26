from typing import Optional



def build_log_message(username: Optional[str] = "anonymous", ip: str = "0.0.0.0", timestamp: str = "1970-01-01 00:00:00") -> str:

    """Builds a log message with the given format: "User {username} logged in with IP {ip} at {timestamp}".



    Args:

        username: The username of the user. Defaults to "anonymous" if not provided.

        ip: The IP address of the user.

        timestamp: The timestamp of the login event.



    Returns:

        A string as a log message.

    """

    return f"User {username} logged in with IP {ip} at {timestamp}"

