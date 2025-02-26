from typing import Union



def convert_topic_name(topic_name: Union[str, bytes]) -> str:

    """Converts a topic name to a collection name.

    Args:

        topic_name: The topic name to convert.

    """

    collection_name = topic_name.replace('-', '_')

    collection_name = collection_name.strip('_')

    collection_name = 'topic_' + collection_name

    return collection_name

