from typing import Dict, Tuple



def collect_and_sort_counts(phone_counts_dict: Dict[str, int]) -> List[Tuple[str, int]]:

    """Collects and sorts the counts of phone numbers in a dictionary.



    Args:

        phone_counts_dict: A dictionary where the keys are the names of people and the values are the counts of phone numbers associated with them.



    Returns:

        A sorted list of tuples, where each tuple contains a name and its corresponding count of phone numbers. The list is sorted in ascending order based on the count of phone numbers.

    """

    phone_counts_list = []

    for name, count in phone_counts_dict.items():

        phone_counts_list.append((name, count))

    phone_counts_list.sort(key=lambda x: x[1])

    return phone_counts_list

