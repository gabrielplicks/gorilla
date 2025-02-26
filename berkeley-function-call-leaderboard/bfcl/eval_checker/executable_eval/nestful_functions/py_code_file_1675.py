from typing import Tuple



def summarize_symbol_table(symbol_table: str) -> str:

    """Summarizes a symbol table into a string of symbols, numbers, and types.



    Args:

        symbol_table: A string representing the symbol table.



    Returns:

        A string summarizing the symbol table.

    """

    summarized_table = ""



    for entry in symbol_table.split('\n'):

        symbol, number, type = entry.split(', ')

        summarized_table += f"{symbol} ({number}), {type}\n"



    return summarized_table

