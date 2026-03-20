"""
DAG - Lazy/DAG approach. No global graph.
Each node has predecessors; Model traverses from output backwards.

Dimensioni: seq, time, dim. Shape = seq + (time,) + dim.
Default: seq=(), time=1, dim=(1,)
"""

# Default dimensioni
SEQ_TIME_DIM_DEFAULT = ((), 1, (1,))


def to_tuple(x, default=(1,)):
    """Converte int/tuple/None in tuple. 0 -> default."""
    if x is None:
        return default
    if isinstance(x, int):
        return (x,) if x != 0 else default
    return tuple(x) if x else default


def get_seq_time_dim(node):
    """Estrae (seq, time, dim) da Stream, Input o Layer. Default: seq=(), time=1, dim=(1,)."""
    seq = tuple(getattr(node, 'seq', ()) or ())
    time = getattr(node, 'time', None) or 1
    dim = to_tuple(getattr(node, 'dim', (1,)), (1,))
    return seq, time, dim


def seq_time_dim_to_shape(seq, time, dim):
    """Converte (seq, time, dim) nella shape Keras (senza batch)."""
    seq = tuple(seq or ())
    dim = to_tuple(dim, (1,))
    return seq + (time,) + dim


def collect_and_order(output_nodes):
    """
    Raccoglie tutti i nodi dagli output e restituisce ordine topologico.
    DFS post-order da output verso input.
    """
    if not isinstance(output_nodes, (list, tuple)):
        output_nodes = [output_nodes]
    result = []
    visited = set()

    def dfs(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        for pred in getattr(node, 'predecessors', []):
            dfs(pred)
        result.append(node)

    for out in output_nodes:
        dfs(out)
    return result


_node_counter = 0


def next_name(prefix: str) -> str:
    """Nome univoco per nodi generati."""
    global _node_counter
    _node_counter += 1
    return f"{prefix}{_node_counter}"
