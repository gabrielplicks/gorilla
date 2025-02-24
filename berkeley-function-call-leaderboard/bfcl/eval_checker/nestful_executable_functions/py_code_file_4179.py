from typing import Dict



def create_widget(widget: Dict[str, str]) -> Dict[str, str]:

    """Creates a dictionary with the following structure:



    {

      'name': 'my_widget',

      'type': 'button',

      'description': 'Press this button',

      'properties': {

        'width': 100,

        'height': 50,

        'color': 'blue'

      }

    }



    Args:

        widget: The dictionary to be updated with the required values.



    Returns:

        The updated dictionary.

    """

    widget['name'] = 'my_widget'

    widget['type'] = 'button'

    widget['description'] = 'Press this button'

    widget['properties'] = {

        'width': 100,

        'height': 50,

        'color': 'blue'

    }

    return widget

