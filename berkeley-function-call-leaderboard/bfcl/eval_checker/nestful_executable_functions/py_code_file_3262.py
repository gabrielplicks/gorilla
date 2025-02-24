from typing import List



def generate_import_statement(module_name: str, imports: List[str]) -> str:

    """Generates an import statement for a given module name and a list of imports.



    Args:

        module_name: The name of the module to import from.

        imports: A list of names to import from the module.

    """

    import_statement = f"from {module_name} import "



    for i, import_name in enumerate(imports):

        if i == len(imports) - 1:

            import_statement += import_name

        else:

            import_statement += f"{import_name}, "



    return import_statement

