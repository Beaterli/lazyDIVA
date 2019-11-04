from pathfinder.finderstate import FinderState


class BiDFSFinder(object):
    def __init__(self, env_graph, max_path_length):
        self.graph = env_graph
        self.max_path_length = max_path_length

    def paths_between(self, from_id, to_id, width=5, path_length=0, id_in_path=None):

        states = []
        if path_length >= self.max_path_length:
            return states

        if id_in_path is None:
            id_in_path = ()

        candidates = self.graph.neighbors_of(from_id)

        for index, neighbor in enumerate(candidates):
            if neighbor.to_id == to_id:
                return [FinderState(
                    path_step=[from_id, to_id]
                )]

        for index, neighbor in enumerate(candidates):
            if len(states) >= width:
                break

            if neighbor.to_id in id_in_path:
                continue

            reverse_states = self.paths_between(
                from_id=to_id,
                to_id=neighbor.to_id,
                width=width,
                path_length=path_length + 1,
                id_in_path=id_in_path + (from_id,)
            )
            if len(reverse_states) == 0:
                continue

            roof = min(width - len(states), len(reverse_states))
            for i in range(0, roof):
                reverse_state = reverse_states[i]
                reverse_state.path.reverse()
                states.append(FinderState(
                    path_step=[from_id],
                    post_state=reverse_state
                ))

        return states
