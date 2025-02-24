import re



def replace_gh_link_with_issue_number(markdown_string: str) -> str:

    """Replaces GitHub issue links in a Markdown string with formatted issue numbers.



    Args:

        markdown_string: The input Markdown string.



    Returns:

        The modified Markdown string with replaced GitHub issue links.

    """

    github_issue_link_pattern = r"https://github.com/[^/]+/[^/]+/issues/(\d+)"

    issue_number_replacement = r"[Issue #\1](https://github.com/username/repo/issues/\1)"



    replaced_string = re.sub(github_issue_link_pattern, issue_number_replacement, markdown_string)



    return replaced_string

