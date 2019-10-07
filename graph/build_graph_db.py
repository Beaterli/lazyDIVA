import os
import sqlite3

dataPath = 'D:\\project\\master-degree\\NELL-995\\'
entityIdFile = dataPath + 'entity2id.txt'
relationIdFile = dataPath + 'relation2id.txt'
entityVecFile = dataPath + 'entity2vec.bern'
relationVecFile = dataPath + 'relation2vec.bern'
graphFile = dataPath + 'kb_env_rl.txt'

if os.path.exists('graph.db'):
    os.remove('graph.db')

conn = sqlite3.connect('graph.db')
conn.execute('''create table entities(
	eid int primary key not null,
	entity text not null,
	emb text not null
)''')
conn.execute('''create table relations(
	rid int primary key not null,
	relation text not null,
	emb text not null
)''')
conn.execute('''create table graph(
	from_id int not null,
	rid int not null,
	to_id int not null
)''')
conn.execute('''create table samples(
	from_id int not null,
	rid int not null,
	to_id int not null,
	"type" text(1) not null,
	"stage" text(5) not null
)''')
conn.commit()

vecFile = open(entityVecFile)
idFile = open(entityIdFile)
entityIds = {}
for idLine in idFile.readlines():
    vecLine = vecFile.readline()
    entity, eid = idLine.split()
    entityIds[entity] = eid
    conn.execute('''insert into entities values (?, ?, ?)''', (eid, entity, vecLine))
vecFile.close()
idFile.close()
conn.commit()

vecFile = open(relationVecFile)
idFile = open(relationIdFile)
relationIds = {}
for idLine in idFile.readlines():
    vecLine = vecFile.readline()
    relation, rid = idLine.split()
    relationIds[relation] = rid
    conn.execute('''insert into relations values (?, ?, ?)''', (rid, relation, vecLine))
vecFile.close()
idFile.close()
conn.commit()

gFile = open(graphFile)
for line in gFile.readlines():
    from_ent, to_ent, relation = line.split()
    conn.execute('''insert into graph values (?, ?, ?)''',
                 (entityIds[from_ent], relationIds[relation], entityIds[to_ent]))
gFile.close()
conn.commit()

for name in os.listdir(dataPath + "tasks"):
    task_path = dataPath + "tasks\\" + name
    if not os.path.isdir(task_path):
        continue

    rel = name.replace("concept_", "concept:")
    trainFile = open(task_path + "\\train.pairs")
    for line in trainFile.readlines():
        from_ent, to_ent = line.split(',')
        from_ent = from_ent.replace('thing$', '')
        to_ent, positive = to_ent.split(': ')
        to_ent = to_ent.replace('thing$', '')
        conn.execute('''insert into samples values(?, ?, ?, ?, ?)''',
                     (entityIds[from_ent], relationIds[rel], entityIds[to_ent], positive, 'train'))
    trainFile.close()

    testFile = open(task_path + "\\test.pairs")
    for line in testFile.readlines():
        from_ent, to_ent = line.split(',')
        from_ent = from_ent.replace('thing$', '')
        to_ent, positive = to_ent.split(': ')
        to_ent = to_ent.replace('thing$', '')
        conn.execute('''insert into samples values(?, ?, ?, ?, ?)''',
                     (entityIds[from_ent], relationIds[rel], entityIds[to_ent], positive, 'test'))
    testFile.close()
conn.commit()

conn.execute('''CREATE UNIQUE INDEX "main"."rid" ON "relations" ("rid" COLLATE BINARY ASC);''')
conn.execute('''CREATE UNIQUE INDEX "main"."eid" ON "entities" ("eid" COLLATE BINARY ASC);''')
conn.commit()

conn.close()
print('finished!')
