from typing import Dict, Any



def parse_json_schema(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:

    """Parses a JSON Schema and returns a dictionary containing all the definitions in the schema.



    Args:

        schema: The JSON Schema to parse.



    Returns:

        A dictionary containing all the definitions in the schema.

    """

    if not isinstance(schema, dict):

        return {}

    if "definitions" in schema:

        definitions = schema["definitions"]

        if isinstance(definitions, dict):

            result = {}

            for name, definition in definitions.items():

                result[name] = parse_json_schema(definition)

            return result

    return {}

