from queue import Queue

from pathfinder.finderstate import FinderState


class BiBFSFinder(object):
    def __init__(self, env_graph, max_path_length):
        self.graph = env_graph
        self.max_path_length = max_path_length

    def cover(self, states, queue, covered_ids, width, reverse=False):
        current_state = queue.get()
        current_id = current_state.path[-1]
        length = len(current_state.path)

        candidates = self.graph.neighbors_of(current_id)

        for index, neighbor in enumerate(candidates):
            if neighbor.to_id in current_state.entities:
                continue

            if neighbor.to_id in covered_ids:
                final_state = FinderState(current_id)
                if reverse:
                    final_state.path = covered_ids[neighbor.to_id].path + current_state.path
                else:
                    final_state.path = current_state.path + covered_ids[neighbor.to_id].path

                states.append(final_state)
                if len(states) == width:
                    return
                else:
                    continue

            elif length < self.max_path_length - 1:
                new_cover = FinderState(
                    path_step=[neighbor.to_id],
                    pre_state=current_state
                )
                queue.put(new_cover)
                covered_ids[neighbor.to_id] = new_cover

    def paths_between(self, from_id, to_id, width=5):
        states = []
        head_queue = Queue()
        tail_queue = Queue()
        head = FinderState(
            path_step=from_id
        )
        tail = FinderState(
            path_step=to_id
        )
        head_queue.put(head)
        tail_queue.put(tail)
        covered_ids = {
            head.path[0]: head,
            tail.path[0]: tail
        }

        while not head_queue.empty() \
                and not tail_queue.empty():
            self.cover(states, head_queue, covered_ids, width, False)
            self.cover(states, tail_queue, covered_ids, width, True)
            if len(states) == width:
                break

        return states
