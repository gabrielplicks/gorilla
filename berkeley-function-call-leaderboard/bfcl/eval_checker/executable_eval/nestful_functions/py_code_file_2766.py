from typing import List



def read_file_and_convert(file_path: str) -> List[List[float]]:

    """Reads a file and returns a list of lists, where each list contains the numbers found in each line.

    The numbers in each line are separated by a space. Each line can have a different number of numbers.

    The function uses string-to-float conversion to convert the numbers to floats.



    Args:

        file_path: The path to the file.

    """

    numbers_list = []



    with open(file_path, 'r') as file:

        for line in file:

            line_numbers = line.split()

            converted_numbers = [float(num) for num in line_numbers]

            numbers_list.append(converted_numbers)



    return numbers_list

