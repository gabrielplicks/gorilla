from typing import Tuple



def create_github_url(username: str, repository: str, file_name: str) -> str:

    """Creates a URL in the format "https://github.com/USERNAME/REPOSITORY/blob/main/FILE.md".



    Args:

        username: The GitHub username.

        repository: The name of the repository.

        file_name: The name of the file.



    Returns:

        A URL in the format "https://github.com/USERNAME/REPOSITORY/blob/main/FILE.md".

    """

    return f"https://github.com/{username}/{repository}/blob/main/{file_name}.md"

