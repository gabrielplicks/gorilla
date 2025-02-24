from typing import List, Tuple



def separate_data(data: List[List[int]]) -> Tuple[List[List[int]], List[int]]:

    """Separates a 2D data matrix into separate lists for the independent variables (X) and the dependent variable (y).



    Args:

        data: A 2D data matrix in the form of a list of lists where each sub-list contains two or more integers.



    Returns:

        A tuple containing the independent variables (X) and the dependent variable (y).

    """

    X = [row[:-1] for row in data]  # Extract the independent variables from each row

    y = [row[-1] for row in data]  # Extract the dependent variable from each row

    return X, y

