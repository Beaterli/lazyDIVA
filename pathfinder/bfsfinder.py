import time

from graph.graph import Graph
from pathfinder.finderstate import FinderState


class BFSFinder(object):
    def __init__(self, env_graph, max_path_length):
        self.graph = env_graph
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
                return [FinderState(
                    path_step=(neighbor.rel_id, to_id),
                    action_chosen=index
                )]

        for index, neighbor in enumerate(candidates):
            if len(states) >= width:
                break

            if neighbor.to_id in id_in_path:
                continue

            postfix_states = self.paths_between(
                from_id=neighbor.to_id,
                to_id=to_id,
                width=width,
                depth=depth + 1,
                id_in_path=id_in_path + (from_id,)
            )
            if len(postfix_states) == 0:
                continue

            roof = min(width - len(states), len(postfix_states))
            for i in range(0, roof):
                states.append(FinderState(
                    path_step=(neighbor.rel_id, neighbor.to_id),
                    action_chosen=index,
                    post_state=postfix_states[i]
                ))

        if depth == 0:
            for state in states:
                state.path = (from_id,) + state.path

        return states


if __name__ == "__main__":
    test_graph_db = 'graph.db'
    graph = Graph(test_graph_db)
    graph.prohibit_relation('concept:athletehomestadium')
    start_time = time.time()
    ep_start = 27414
    ep_end = 59928
    paths = BFSFinder(graph, 5).paths_between(from_id=ep_start, to_id=ep_end, width=5)
    print(str(paths))
    print('from {} to {} takes {}s'.format(ep_start, ep_end, time.time() - start_time))
