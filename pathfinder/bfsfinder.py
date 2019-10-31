from finderstate import FinderState

from graph.graph import Graph


class BFSFinder(object):
    def __init__(self, graph, max_path_length):
        self.graph = graph
        self.max_path_length = max_path_length

    def paths_between(self, from_id, to_id, width=5, depth=0, id_in_path=None):
        states = []
        if depth == self.max_path_length:
            return states

        if id_in_path is None:
            id_in_path = ()

        candidates = []
        for neighbor in self.graph.neighbors_of(from_id):
            if neighbor.to_id not in id_in_path:
                candidates.append(neighbor)

        for index, neighbor in enumerate(candidates):
            if neighbor.to_id == to_id:
                return FinderState(
                    path_step=(neighbor.rel_id, to_id),
                    action_chosen=index
                )

        for index, neighbor in enumerate(candidates):
            if len(states) == width:
                return states

            if neighbor.to_id in id_in_path:
                continue

            sub_states = self.paths_between(neighbor.to_id, to_id, depth + 1, id_in_path + (from_id,))
            if len(sub_states) == 0:
                continue

            header = ()
            if depth == 0:
                header = (from_id,)
            for sub_state in sub_states:
                states.append(FinderState(
                    path_step=header + (neighbor.rel_id, neighbor.to_id) + sub_state.path,
                    action_chosen=index,
                    next_state=sub_state
                ))

        return states


if __name__ == "__main__":
    test_graph_db = 'graph.db'
    graph = Graph(test_graph_db)
    graph.prohibit_relation('concept:athletehomestadium')
    print(str(BFSFinder(graph, 10).paths_between(1245, 19623, 5)))
