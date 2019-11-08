import sqlite3

conn = sqlite3.connect('graph.db')
ent_file = open('entity2id.txt', 'w')
rel_file = open('relation2id.txt', 'w')
pair_file = open('train2id.txt', 'w')

ent_count = 0
for row in conn.execute('''select count(eid) from entities'''):
    ent_count = row[0]

ent_file.write(str(ent_count) + '\n')
for (entity, eid) in conn.execute('''select entity, eid from entities'''):
    ent_file.write('{}\t{}\n'.format(entity, eid))
ent_file.close()

rel_count = 0
for row in conn.execute('''select count(rid) from relations'''):
    rel_count = row[0]

rel_file.write(str(rel_count) + '\n')
for (relation, rid) in conn.execute('''select relation, rid from relations'''):
    rel_file.write('{}\t{}\n'.format(relation, rid))
rel_file.close()

pair_count = 0
for row in conn.execute('''select count(*) from graph'''):
    pair_count = row[0]

pair_file.write(str(pair_count) + '\n')
for (from_id, rid, to_id) in conn.execute('''select from_id, rid, to_id from graph'''):
    pair_file.write('{} {} {}\n'.format(from_id, to_id, rid))
pair_file.close()

print('finished!')
