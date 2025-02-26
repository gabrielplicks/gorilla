from typing import List



def ssd(img1: List[List[int]], img2: List[List[int]]) -> int:

    """Calculates the sum of squared differences between two images.



    Args:

        img1: A two-dimensional list representing the first image.

        img2: A two-dimensional list representing the second image.

    """

    ssd_sum = 0

    for i in range(len(img1)):

        for j in range(len(img1[i])):

            ssd_sum += (img1[i][j] - img2[i][j])**2



    return ssd_sum

