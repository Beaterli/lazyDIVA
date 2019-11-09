import time

from graph.graph import Graph
from pathfinder.brute.bfsfinder import BFSFinder
from pathfinder.brute.dfsfinder import DFSFinder

test_graph_db = 'graph.db'
graph = Graph(test_graph_db)
graph.prohibit_relation('concept:athletehomestadium')
start_time = time.time()
ep_start = 2592
ep_end = 62111

print('DFS Finder:')
paths = DFSFinder(graph, 5).paths_between(from_id=ep_start, to_id=ep_end, width=5)
print(str(paths))
print('takes {}s'.format(time.time() - start_time))

print('BFS Finder:')
paths = BFSFinder(graph, 5).paths_between(from_id=ep_start, to_id=ep_end, width=5)
print(str(paths))
print('takes {}s'.format(time.time() - start_time))

