from typing import List, Union



def get_table_cells(table: List[List[Union[str, None]]]) -> List[str]:

    """Returns a list of strings where each string is the text of a table cell.

    If a cell is empty, a placeholder string ("N/A") is added instead.

    The table is represented with a list of lists.



    Args:

        table: A list of lists representing a table.



    Returns:

        A list of strings where each string is the text of a table cell.

    """

    cells = []

    for row in table:

        for cell in row:

            if cell:  # Check if the cell is not empty

                cells.append(cell)  # Add the cell text to the list of cells

            else:

                cells.append("N/A")  # Add a placeholder string if the cell is empty



    return cells

