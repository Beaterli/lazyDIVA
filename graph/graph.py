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
        self.entEmb = [np.zeros(1)] * ent_count

        rel_count = 0
        for row in self.conn.execute('''select count(rid) from relations'''):
            rel_count = row[0]
        self.relEmb = [np.zeros(1)] * rel_count

        self.prohibits = []

        self.neighbors = [[]] * len(self.entEmb)
        for i in range(len(self.neighbors)):
            self.neighbors[i] = []
        for row in self.conn.execute('''select from_id, rid, to_id from graph''').fetchall():
            self.neighbors[row[0]].append(Link(row[1], row[2]))
        print('graph load complete!')

    def samples_of(self, relation_text, stage):
        samples = []
        for row in self.conn.execute('''select from_id, to_id, "type" from samples 
            where rid = (select rid from relations where relation = ?) and stage = ?''',
                                     (relation_text, stage)):
            samples.append({
                'from_id': row[0],
                'to_id': row[1],
                'type': row[2]
            })
        return samples

    def train_samples_of(self, relation_text):
        return self.samples_of(relation_text, "train")

    def test_samples_of(self, relation_text):
        return self.samples_of(relation_text, "test")

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
        if self.entEmb[ent_id].size == 1:
            for row in self.conn.execute('''select emb from entities where eid = {}'''.format(ent_id)).fetchall():
                self.entEmb[ent_id] = np.genfromtxt(StringIO(row[0]), dtype="f4")
        return self.entEmb[ent_id]

    def vec_of_rel(self, rel_id):
        if self.relEmb[rel_id].size == 1:
            for row in self.conn.execute('''select emb from relations where rid = {}'''.format(rel_id)).fetchall():
                self.relEmb[rel_id] = np.genfromtxt(StringIO(row[0]), dtype="f4")
        return self.relEmb[rel_id]

    def vec_of_rel_name(self, rel_name):
        vec = np.zeros(200)
        for row in self.conn.execute('''select emb from relations where relation = ?''', rel_name):
            vec = np.genfromtxt(StringIO(row[0]), dtype="f4")
        return vec

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
        return "{} -> {}".format(self.rel_id, self.to_id)

    __repr__ = __str__
