from typing import Union



def get_origin_name(repo_url: str, branch_name: str) -> str:

    """

    Determines the remote origin name based on the given Git repository URL and branch name.

    Args:

        repo_url: The URL of the Git repository.

        branch_name: The name of the branch.

    """

    if repo_url.lower().endswith(".git") and branch_name.lower() == "master":

        return "origin"

    elif not repo_url.lower().endswith(".git") and branch_name.lower() == "master":

        return "github"

    elif branch_name.lower() != "master" and repo_url.lower().endswith(".git"):

        return "git"

    else:

        return "github"

