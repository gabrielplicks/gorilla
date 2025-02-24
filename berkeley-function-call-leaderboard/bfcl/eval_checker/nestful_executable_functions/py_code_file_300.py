import re

from typing import List



def get_board_ids(table_names: List[str]) -> List[str]:

    """

    Update the `get_board_ids` function to get a list of board IDs from given table names

    according to a specific naming convention. The naming convention is that each table name

    represents a board, and the board ID is the table name without the prefix 'board_'.

    Args:

        table_names: A list of table names.

    """

    board_ids = []

    pattern = re.compile(r'^board_(\w+)')



    for table_name in table_names:

        match = pattern.match(table_name)

        if match:

            board_id = match.group(1)

            board_ids.append(board_id)



    return board_ids

