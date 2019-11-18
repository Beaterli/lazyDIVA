from queue import Queue

from pathfinder.finderstate import FinderState

max_queue_size = 4 * 1024 * 1024


class BFSFinder(object):
    def __init__(self, env_graph, max_path_length):
        self.graph = env_graph
        self.max_path_length = max_path_length

    def paths_between(self, from_id, to_id, width=5):
        states = []
        queue = Queue()
        queue.put(FinderState(
            path_step=from_id
        ))

        while not queue.empty():
            if queue.qsize() > max_queue_size:
                print('ended search due to large queue')
                break

            current_state = queue.get()
            current_id = current_state.path[-1]
            current_length = len(current_state.path)

            candidates = self.graph.neighbors_of(current_id)

            for index, neighbor in enumerate(candidates):
                if neighbor.to_id == to_id:
                    states.append(FinderState(
                        path_step=neighbor.to_list(),
                        pre_state=current_state
                    ))
                    if len(states) == width:
                        return states
                    else:
                        continue
                elif current_length < self.max_path_length * 2 \
                        and neighbor.to_id not in current_state.entities:
                    queue.put(FinderState(
                        path_step=neighbor.to_list(),
                        pre_state=current_state
                    ))

        return states
