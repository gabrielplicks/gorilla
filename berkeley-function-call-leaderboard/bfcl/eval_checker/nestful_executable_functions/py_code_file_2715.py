from typing import Tuple



def get_branch_url(repo: str, branch: str) -> str:

    """Constructs a URL to a specific branch of a GitHub repo.



    Args:

        repo: The name of the GitHub repo.

        branch: The name of the branch.



    Returns:

        A URL string that follows the format `https://github.com/<repo>/tree/<branch>`.

    """

    return f"https://github.com/{repo}/tree/{branch}"

