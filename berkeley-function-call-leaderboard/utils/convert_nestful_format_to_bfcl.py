import json

TYPE_MAPPING = {
    "str": "string",
    "int": "integer",
    "float": "float",
    "bool": "boolean",
    "any": "any",
    "dict": "dict",
    "list": "array",
    "tuple": "tuple",
}

# Paths
nestful_dataset_path = "data_v2/nestful_data.jsonl"
nestful_to_bfcl_dataset_path = "BFCL_v3_exec_parallel_multiple_nestful.json"


def type_list_to_one_type(parameter_type):
    if parameter_type == ["integer", "number"]:
        parameter_type = "int"
    elif parameter_type == ["number", "string"]:
        parameter_type = "float"
    elif parameter_type == ["object", "string"]:
        parameter_type = "string"
    elif parameter_type == ["integer", "number", "string"]:
        parameter_type = "int"
    elif parameter_type == ["integer", "string"]:
        parameter_type = "int"
    elif parameter_type == ["boolean", "string"]:
        parameter_type = "bool"
    elif parameter_type == ["array", "object"]:
        parameter_type = "list"
    elif parameter_type == ["integer", "object"]:
        parameter_type = "int"
    elif parameter_type == ["boolean", "integer"]:
        parameter_type = "bool"
    elif parameter_type == ["array", "string"]:
        parameter_type = "list"
    elif parameter_type == ["object", "object"]:
        parameter_type = "string"
    elif parameter_type == ["array", "object", "string"]:
        parameter_type = "string"
    elif parameter_type == ["integer", "number", "object", "string"]:
        parameter_type = "float"
    elif parameter_type == ["boolean", "integer", "string"]:
        parameter_type = "string"
    elif parameter_type == ["integer", "number", "object"]:
        parameter_type = "float"
    elif parameter_type == ["array", "array"]:
        parameter_type = "list"
    elif parameter_type == ["array", "integer", "number", "string"]:
        parameter_type = "string"
    elif parameter_type == ["boolean", "integer", "number", "object", "string"]:
        parameter_type = "float"
    elif parameter_type == ["array", "integer", "string"]:
        parameter_type = "string"
    elif parameter_type == ["array", "object", "object"]:
        parameter_type = "string"
    elif parameter_type == ["object", "object", "string"]:
        parameter_type = "string"
    elif parameter_type == ["array", "integer"]:
        parameter_type = "string"
    return parameter_type


# Load the NESTful data
nestful_data = []
with open(nestful_dataset_path, "r") as f:
    for line in f:
        nestful_data.append(json.loads(line))

# Transform each entry in the NESTful data to the BFCL format
# "sample_id": some unique id -> "id": "nestful_" + str(entry_number)
# "input": str question text -> "question": [{"role": "user", "content": str question text}]
# "output" -> "ground_truth"
# "tools" -> "function"
# "gold_answer" -> "gold_answer"
nestful_to_bfcl_dataset = []
item_to_check = 48
for i, entry in enumerate(nestful_data):
    if i == item_to_check:
        print("BEFORE:")
        print(json.dumps(entry))
    nestful_to_bfcl_entry = {
        "id": "exec_parallel_multiple_nestful_" + str(i),
        "question": [[{"role": "user", "content": entry["input"]}]],
        "ground_truth": entry["output"],
        "gold_answer": entry["gold_answer"],
    }
    # Functions need to have the following format:
    # {
    #     "name": entry_tool["name"],
    #     "description": entry_tool["description"],
    #     "parameters": {
    #         "type": "dict",
    #         "properties": {
    #             "property_name" (entry_parameter["name"]): {
    #                 "type": TYPE_MAPPING[entry_parameter["type"]],
    #                 "description": entry_parameter["description"],
    #             }
    #         },
    #         "required": all property names,
    #     }
    # }
    functions = []
    for entry_tool in entry["tools"]:
        parameters = {
            "type": "dict",
            "properties": {},
            "required": [],
        }
        if "properties" in entry_tool["parameters"]:
            entry_tool["parameters"] = entry_tool["parameters"]["properties"]
        for entry_parameter_name, entry_parameter_data in entry_tool["parameters"].items():
            if "type" not in entry_parameter_data:
                entry_parameter_data["type"] = "string"
            if entry_parameter_data["type"] == "int or float":
                entry_parameter_data["type"] = "float"
            parameter_type = entry_parameter_data["type"]
            if isinstance(parameter_type, list):
                parameter_type = type_list_to_one_type(parameter_type)
            parameter_type = TYPE_MAPPING.get(parameter_type, parameter_type)
            parameters["properties"][entry_parameter_name] = {
                "type": parameter_type,
                "description": entry_parameter_data["description"],
            }

            # If the parameter type is an array, we need to add the "items" key
            if parameter_type == "array":
                if i == item_to_check:
                    print("\nBEFORE:")
                    print(json.dumps(entry_tool["parameters"]))

                items = entry_parameter_data.get("items", None)
                if items:
                    # Get items type
                    items_dict = entry_parameter_data["items"]
                    # Sometimes the items type is a list
                    if isinstance(items_dict["type"], list):
                        print("Entered isinstance(items_dict['type'], list)")
                        items_dict["type"] = type_list_to_one_type(items_dict["type"])
                        print("items_dict['type']=", items_dict["type"])

                    # If the items type is an array, there should be a nested items key
                    if items_dict["type"] in ["array", "list"]:
                        print()
                        print("i=", i)
                        # Check if it is specifying an "items" key or a "prefixItems" key
                        if "prefixItems" in items_dict:
                            print("Entered prefixItems")
                            subitems_dict = items_dict["prefixItems"]
                        elif "items" in items_dict:
                            print("Entered items")
                            subitems_dict = items_dict["items"]
                            print(subitems_dict)
                        else:
                            print("Entered else type: string")
                            subitems_dict = {"type": "string"}
                        # If the subitems is a list (cause it was a tuple)
                        if isinstance(subitems_dict, list):
                            print("Entered isinstance(subitems_dict, list)")
                            subitems_dict = subitems_dict[0]
                            # Sometimes the dict at position 0 is empty
                            if not subitems_dict:
                                print("Entered not subitems_dict")
                                subitems_dict = {"type": "string"}
                            # Sometimes the items type is a list
                            if isinstance(subitems_dict["type"], list):
                                print("Entered isinstance(subitems_dict['type'], list)")
                                subitems_dict["type"] = type_list_to_one_type(subitems_dict["type"])
                                print("NOW IS subitems_dict['type']=", subitems_dict["type"])

                        if not subitems_dict:
                            subitems_dict = {"type": "string"}

                        if i == item_to_check:
                            print("\nSUBITEMS:")
                            print(subitems_dict)

                        # Sometimes the items type is a list
                        if isinstance(subitems_dict["type"], list):
                            print("Entered isinstance(subitems_dict['type'], list)")
                            print("WAS subitems_dict['type']=", subitems_dict["type"])
                            subitems_dict["type"] = type_list_to_one_type(subitems_dict["type"])
                            print("NOW IS subitems_dict['type']=", subitems_dict["type"])

                        # Check if type name is good
                        if subitems_dict["type"] not in TYPE_MAPPING.values():
                            if subitems_dict["type"] == "number":
                                subitems_dict["type"] = "float"
                            elif subitems_dict["type"] == "object":
                                subitems_dict["type"] = "dict"
                            subitems_dict["type"] = TYPE_MAPPING[subitems_dict["type"]]
                        print("subitems_dict['type']=", subitems_dict["type"])

                        if i == item_to_check:
                            print("\nSUBITEMS AFTER:")
                            print(subitems_dict)

                        items_dict = {"type": "array", "items": subitems_dict}
                    else:
                        if items_dict["type"] not in TYPE_MAPPING.values():
                            if items_dict["type"] == "number":
                                items_dict["type"] = "float"
                            elif items_dict["type"] == "object":
                                items_dict["type"] = "dict"
                            items_dict["type"] = TYPE_MAPPING[items_dict["type"]]
                else:
                    items_dict = {"type": "string"}

                parameters["properties"][entry_parameter_name]["items"] = items_dict

                if i == item_to_check:
                    print("\nAFTER:")
                    print(json.dumps(parameters))

        parameters["required"] = list(parameters["properties"].keys())
        functions.append(
            {
                "name": entry_tool["name"],
                "description": entry_tool["description"],
                "parameters": parameters,
            }
        )
    nestful_to_bfcl_entry["function"] = functions
    if i == item_to_check:
        print("\nFINAL AFTER:")
        print(json.dumps(nestful_to_bfcl_entry))
    nestful_to_bfcl_dataset.append(nestful_to_bfcl_entry)

# Save the NESTful to BFCL data
with open(nestful_to_bfcl_dataset_path, "w") as f:
    for entry in nestful_to_bfcl_dataset:
        f.write(json.dumps(entry) + "\n")
