from pathlib import Path



def get_home_path(github_account_name: str) -> Path:

    """Returns the absolute path to the user's home directory for a given GitHub account name.



    Args:

        github_account_name: The GitHub account name.

    """

    return Path.home() / github_account_name

