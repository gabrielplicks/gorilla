from typing import List, Tuple



def print_names_and_ages(seq: List[Tuple[str, int]]) -> None:

    """Prints the list of names and ages in the specified format.



    Args:

        seq: A sequence of tuples of the form (name, age).

    """

    print("Name\tAge")

    for name, age in seq:

        print(f"{name}\t{age}")

