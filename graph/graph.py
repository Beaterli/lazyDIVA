class Graph(object):
    def __init__(self):
        self.nodes = {}

    def add_relation(self, from_node, edge, to_node):
        if from_node in self.nodes:
            self.nodes[from_node].append(Link(edge, to_node))
        else:
            self.nodes[from_node] = [Link(edge, to_node)]

    def get_links_from(self, node):
        return self.nodes[node]

    def remove_links_between(self, from_node, to_node):
        for idx, link in enumerate(self.nodes[from_node]):
            if link.to_node == to_node:
                del self.nodes[from_node][idx]
                break
        for idx, link in enumerate(self.nodes[to_node]):
            if link.to_node == from_node:
                del self.nodes[to_node][idx]
                break

    def random_nodes_between(self, from_node, to_node, num):
        import random

        res = set()
        if num > len(self.nodes) - 2:
            raise ValueError('Number of Intermediates picked is larger than possible',
                             'num_entities: {}'.format(len(self.nodes)), 'num_itermediates: {}'.format(num))
        for i in range(num):
            intermediate = random.choice(self.nodes.keys())
            while intermediate in res or intermediate == from_node or intermediate == to_node:
                intermediate = random.choice(self.nodes.keys())
            res.add(intermediate)
        return list(res)

    def __str__(self):
        string = ""
        for entity in self.nodes:
            string += entity + ','.join(str(x) for x in self.nodes[entity])
            string += '\n'
        return string


class Link(object):
    def __init__(self, edge, to_node):
        self.edge = edge
        self.to_node = to_node

    def __str__(self):
        return "\t{}\t{}".format(self.edge, self.to_node)

    __repr__ = __str__
