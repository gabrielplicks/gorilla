from typing import List, Dict, Any



def combine_gathered_data(gathered_data: List[List[Any]], process_ids: List[int]) -> List[Dict[str, Any]]:

    """Combines the data from multiple processes into a uniform structure.



    Args:

        gathered_data: The data gathered from each process.

        process_ids: The IDs of the processes.



    Returns:

        A list of dictionaries with the required structure.

    """

    combined_data = []

    for i, chunk in enumerate(gathered_data):

        process_id = process_ids[i]

        combined_data.append({

            'process_id': process_id,

            'data': chunk

        })



    return combined_data

