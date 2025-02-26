import dis

from typing import Any



def compile_ast(ast: Any) -> dis.Bytecode:

    """Compiles an abstract syntax tree (AST) to Python bytecode.



    Args:

        ast: The abstract syntax tree to compile.



    Returns:

        The compiled bytecode object.

    """

    source_code = compile(ast, '<string>', 'exec')

    bytecode = dis.Bytecode(source_code)

    return bytecode

