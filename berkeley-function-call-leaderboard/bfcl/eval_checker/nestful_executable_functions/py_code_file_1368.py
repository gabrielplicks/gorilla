from typing import List, Dict



def generate_commit_messages(commits: List[Dict], branch_name: str) -> List[str]:

    """Generates commit messages for a list of commits, with a branch name included at the beginning.



    Args:

        commits: A list of dictionaries. Each commit has a key 'message' and an optional key 'pull_request_number'.

        branch_name: A string representing the branch name.



    Returns:

        A list of strings where each string is the message with the branch name as the initial part. If the commit has a pull request number, append the pull request number to the message in the format '(PR #number)'.

    """

    commit_messages = []

    for commit in commits:

        message = branch_name + ': ' + commit['message']

        if 'pull_request_number' in commit:

            message += f' (PR #{commit["pull_request_number"]})'

        commit_messages.append(message)

    return commit_messages

