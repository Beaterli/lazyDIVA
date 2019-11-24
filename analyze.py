from __future__ import absolute_import, division, print_function
# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf

import checkpoints as chk
from graph.graph import Graph
from pathfinder.lstmfinder import LSTMFinder
from pathreasoner.graph_sage_reasoner import GraphSAGEReasoner
from test_tools import predict_sample, predict_to_label

import pymysql
import json

emb_size = 100
beam = 5
max_path_length = 5

graph = Graph('weibo.db')
graph.prohibit_relation('entertainment')
graph.prohibit_relation('political')
rel_embs = {
    10: graph.vec_of_rel_name('entertainment'),
    12: graph.vec_of_rel_name('political')
}

checkpoint_dir = 'checkpoints/weibo/event_type/unified/sage/'

prior = LSTMFinder(graph=graph, emb_size=emb_size, max_path_length=max_path_length, prior=True)
path_reasoner = GraphSAGEReasoner(graph=graph, emb_size=emb_size, neighbors=15)
path_reasoner_name = type(path_reasoner).__name__
print('using {}, {}'.format(path_reasoner_name, type(prior).__name__))

prior_checkpoint = tf.train.Checkpoint(model=prior)

likelihood_checkpoint = tf.train.Checkpoint(model=path_reasoner)

chk.load_from_index(
    prior_checkpoint,
    checkpoint_dir, 'prior',
    5
)

chk.load_from_index(
    likelihood_checkpoint,
    checkpoint_dir, path_reasoner_name,
    5
)

system_db = pymysql.connect("localhost", "root", "experimental", "weibo")
roots_cursor = system_db.cursor()
roots = []
roots_cursor.execute(
    '''select root.mid, blog.mid child from root, blog
where root.depth > 4 
and root.`type` is null
and root.mid = blog.root_id
and blog.stage = 5
GROUP BY root.mid''')
rows = roots_cursor.fetchall()
for row in rows:
    roots.append({
        'mid': row[0],
        'child': row[1]
    })
roots_cursor.close()
print('analyzing {} roots'.format(len(roots)))


def path_to_ent(path):
    entities = []
    for i in range(0, len(path), 2):
        entities.append(graph.id_to_ent(path[i]))
    return entities


updates = []
for root in roots:
    from_id = graph.mid_to_id(root['mid'])
    to_id = graph.mid_to_id(root['child'])
    if from_id == -1 or to_id == -1:
        print('unrecognized root: {}'.format(root['mid']))
        continue

    predict, paths = predict_sample(
        sample={
            'from_id': from_id,
            'to_id': to_id
        },
        finder=prior,
        beam=beam,
        reasoner=path_reasoner,
        check_dest=False
    )
    label = predict_to_label(predict)
    ent_paths = list(map(lambda p: path_to_ent(p), paths))
    print('blog {} label: {}'.format(root['mid'], label))
    updates.append([label, json.dumps(ent_paths), root['mid']])

update_cur = system_db.cursor()
update_cur.executemany('''update root set `type` = %s, paths = %s where mid = %s''', updates)
system_db.commit()
update_cur.close()

print('analyze complete!')
