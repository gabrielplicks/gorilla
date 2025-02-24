from typing import IO



def create_logger_file(input_file: str) -> None:

    """Creates a logger file with the sum of all integers in the input file, and the conversion of the first number on the first and second lines from integer to string.



    Args:

        input_file: The input file containing integer data.

    """

    with open(input_file, "r") as reader:

        lines = reader.readlines()



    with open("logger.out", "w") as writer:

        writer.write(f"{sum(int(line) for line in lines)}\n")

        writer.write(f"{str(int(lines[0]))}\n")

        writer.write(f"{str(int(lines[1]))}\n")

