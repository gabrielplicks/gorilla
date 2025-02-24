from typing import List, Tuple



def construct_syntax_tree(tokenized_data: List[Tuple[str, str]]) -> dict:

    """Constructs a syntax tree from a given tokenized data.



    Args:

        tokenized_data: A list of tokenized data in the form of a tuple, where the first item is the token and the second item is the token's context.



    Returns:

        A syntax tree with context-aware parsing.

    """

    tree = {}

    for token, context in tokenized_data:

        if context not in tree:

            tree[context] = []

        tree[context].append(token)



    return tree

