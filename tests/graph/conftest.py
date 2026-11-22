import pytest


def _bfs_order(adjacency, source):
    if source not in adjacency:
        return []
    order, discovered, queue = [], {source}, [source]
    while queue:
        vertex = queue.pop(0)
        order.append(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in discovered:
                discovered.add(neighbor)
                queue.append(neighbor)
    return order


def _dfs_order(adjacency, source):
    order = []

    def visit(vertex):
        order.append(vertex)
        for neighbor in adjacency[vertex]:
            if neighbor not in order:
                visit(neighbor)

    if source in adjacency:
        visit(source)
    return order


@pytest.fixture
def reference_orders():
    return _bfs_order, _dfs_order
