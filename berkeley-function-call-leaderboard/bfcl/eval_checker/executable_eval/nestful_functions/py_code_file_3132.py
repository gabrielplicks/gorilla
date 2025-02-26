from typing import List, Dict



def format_counts(counts: List[Dict[str, int]]) -> str:

    """Formats the names of all dictionaries where the corresponding count is greater than 1.



    Args:

        counts: A list of dictionaries, each containing a `name` and `count` key.



    Returns:

        A string containing the `name`s of all dictionaries where the corresponding `count` is greater than 1,

        separated by commas, and each `name` should be enclosed in double quotes, e.g. `"foo", "bar"`

    """

    result = []

    for count in counts:

        if count["count"] > 1:

            result.append(f'"{count["name"]}"')

    return ", ".join(result)

