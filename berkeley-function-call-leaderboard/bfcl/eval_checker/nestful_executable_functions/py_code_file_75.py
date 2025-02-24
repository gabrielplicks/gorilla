from typing import List



def sudoku_to_csv(input: str) -> str:

    """Converts a sudoku board from a string to a CSV string.



    Args:

        input: A string representing a sudoku board. Rows are separated by the newline character "\n", columns are separated by the space character " ", and empty cells are represented by the character "0".



    Returns:

        A string representing the sudoku board in CSV format.

    """

    rows: List[str] = input.split("\n")

    csv_rows: List[str] = []

    for row in rows:

        columns: List[str] = row.split(" ")

        csv_row: str = ",".join([str(int(col)) for col in columns])

        csv_rows.append(csv_row)

    return "\n".join(csv_rows)

