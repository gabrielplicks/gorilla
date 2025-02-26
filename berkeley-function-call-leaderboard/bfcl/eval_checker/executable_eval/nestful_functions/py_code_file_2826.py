from typing import Dict



def compute_frequency(file_path: str) -> Dict[int, int]:

    """Reads a data file containing a list of integers and returns a dictionary containing the frequency of each integer.



    Args:

        file_path: The path to the data file.



    Returns:

        A dictionary containing the frequency of each integer.

    """

    with open(file_path, "r") as file:

        freq_dict = {}

        line = file.readline()



        while line:

            line = line.strip()

            if line:

                num = int(line)

                freq_dict[num] = freq_dict.get(num, 0) + 1



            line = file.readline()



    return freq_dict

