import sqlite3
from io import StringIO

import numpy as np


class Graph(object):
    def __init__(self, sqlite_file):
        self.conn = sqlite3.connect(sqlite_file)
        self.conn.execute('''pragma cache_size = 131072''')
        print('opened database: ', sqlite_file)

        ent_count = 0
        for row in self.conn.execute('''select count(eid) from entities'''):
            ent_count = row[0]
        self.entEmb = [np.array] * ent_count
        print('loading {} entities...'.format(ent_count))
        for row in self.conn.execute('''select eid, emb from entities''').fetchall():
            self.entEmb[row[0]] = np.genfromtxt(StringIO(row[1]))
        print('entity embedding load complete!')

        rel_count = 0
        for row in self.conn.execute('''select count(rid) from relations'''):
            rel_count = row[0]
        self.relEmb = [np.array] * rel_count
        print('loading {} relations...'.format(rel_count))
        for row in self.conn.execute('''select rid, emb from relations''').fetchall():
            self.relEmb[row[0]] = (np.genfromtxt(StringIO(row[1])))
        print('relation embedding load complete!')

        self.prohibits = []

        self.neighbors = [[]] * len(self.entEmb)
        for i in range(len(self.neighbors)):
            self.neighbors[i] = []
        for row in self.conn.execute('''select from_id, rid, to_id from graph''').fetchall():
            self.neighbors[row[0]].append(Link(row[1], row[2]))
        print('graph load complete!')

    def prohibit_relation(self, relation_text):
        self.prohibits.clear()
        for row in self.conn.execute('''select rid from relations where relation = ? or relation = ?''',
                                     (relation_text, relation_text + '_inv')).fetchall():
            self.prohibits.append(row[0])

    def neighbors_of(self, ent_id):
        neighbors = []
        for _, neighbor in enumerate(self.neighbors[ent_id]):
            if neighbor.rel_id not in self.prohibits:
                neighbors.append(neighbor)
        return neighbors

    def vec_of_ent(self, ent_id):
        return self.entEmb[ent_id]

    def vec_of_rel(self, rel_id):
        return self.relEmb[rel_id]

    def random_nodes_between(self, from_node_id, to_node_id, num):
        import random

        res = set()
        if num > len(self.neighbors) - 2:
            raise ValueError('Number of Intermediates picked is larger than possible',
                             'num_entities: {}'.format(len(self.neighbors)), 'num_itermediates: {}'.format(num))
        for i in range(num):
            intermediate = random.randint(0, len(self.neighbors) - 1)
            while intermediate in res or intermediate == from_node_id or intermediate == to_node_id:
                intermediate = random.randint(0, len(self.neighbors) - 1)
            res.add(intermediate)
        return list(res)

    def random_edge_of(self, ent_id):
        import random
        return random.choice(self.neighbors[ent_id])

    def __str__(self):
        string = ""
        for entId in range(len(self.neighbors) - 1):
            string += str(entId) + ','.join(str(x) for x in self.neighbors[entId])
            string += '\n'
        return string

    def __del__(self):
        self.conn.close()


class Link(object):
    def __init__(self, rel_id, to_id):
        self.rel_id = rel_id
        self.to_id = to_id

    def __str__(self):
        return " {} -> {}".format(self.rel_id, self.to_id)

    __repr__ = __str__
