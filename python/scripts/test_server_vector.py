import ast
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parents[1] / "usearch" / "server.py"


def _one_vector():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_one_vector")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(SERVER), "exec"), namespace)
    return namespace["_one_vector"]


def test_single_server_vector_is_one_row():
    one_vector = _one_vector()
    flat = np.arange(4, dtype=np.float32)

    prepared = one_vector(flat)
    assert prepared.shape == (4,)
    assert prepared.reshape(1, -1).shape == (1, 4)
    assert np.array_equal(prepared, flat)

    assert one_vector(flat.reshape(1, 4)).shape == (4,)
    assert one_vector(flat.reshape(4, 1)).shape == (4,)


def test_server_handlers_do_not_turn_a_vector_into_a_column():
    source = SERVER.read_text(encoding="utf-8")
    assert "reshape(vector.shape[0], 1)" not in source
    assert "index.add(int(key), _one_vector(vector))" in source
    assert "index.search(_one_vector(vector), count)" in source
