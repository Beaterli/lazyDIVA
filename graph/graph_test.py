from graph import Graph

test_graph_db = 'graph.db'
graph = Graph(test_graph_db)
graph.prohibit_relation('concept:organizationhiredperson')
print(graph.neighbors_of(2))
