from typing import Dict, List



def combine_labels(label_dict_list: List[Dict[int, List[str]]]) -> Dict[int, List[str]]:

    """Combines two labeling dictionaries into a single dictionary.



    Args:

        label_dict_list: A list of two dictionaries, where each dictionary is a mapping from a tag ID to a list of tags.



    Returns:

        A single dictionary, where each key is a tag ID and the value is a list of tags, with the tags from each of the two dictionaries concatenated together.

    """

    label_dict = {}

    for label_dict_idx in range(len(label_dict_list)):

        for tag_id, tags in label_dict_list[label_dict_idx].items():

            if tag_id in label_dict:

                label_dict[tag_id] += tags

            else:

                label_dict[tag_id] = tags

    return label_dict

