# coding=utf-8
import os
import sqlite3

dataPath = 'D:\\project\\master-degree\\weibo\\'
entityIdFile = dataPath + 'entity2id.txt'
relationIdFile = dataPath + 'relation2id.txt'
entityVecFile = dataPath + 'entity2vec.bern'
relationVecFile = dataPath + 'relation2vec.bern'
graphFile = dataPath + 'train2id.txt'

if os.path.exists('weibo.db'):
    os.remove('weibo.db')

conn = sqlite3.connect('weibo.db')
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
	to_id int not null,
	rid int not null,
	stage text(5) not null
)''')
conn.commit()

vecFile = open(entityVecFile)
idFile = open(entityIdFile, encoding='UTF-8')
entityIds = {}
for idLine in idFile.readlines():
    vecLine = vecFile.readline()
    entity, eid = idLine.split('\t')
    conn.execute('''insert into entities values (?, ?, ?)''', (eid, entity, vecLine))
vecFile.close()
idFile.close()
conn.commit()

vecFile = open(relationVecFile)
idFile = open(relationIdFile, encoding='UTF-8')
relationIds = {}
for idLine in idFile.readlines():
    vecLine = vecFile.readline()
    relation, rid = idLine.split('\t')
    conn.execute('''insert into relations values (?, ?, ?)''', (rid, relation, vecLine))
vecFile.close()
idFile.close()
conn.commit()

gFile = open(graphFile)
for line in gFile.readlines():
    from_ent, to_ent, relation = line.split('\t')
    conn.execute('''insert into graph values (?, ?, ?)''',
                 (from_ent, relation, to_ent))
gFile.close()
conn.commit()

trainFile = open(dataPath + "train.pairs")
for line in trainFile.readlines():
    from_id, to_id, rid = line.split('\t')

    conn.execute('''insert into samples values(?, ?, ?, ?)''',
                 (from_id, to_id, rid,
                  'train'))
trainFile.close()

testFile = open(dataPath + "test.pairs")
for line in testFile.readlines():
    from_id, to_id, rid = line.split('\t')

    conn.execute('''insert into samples values(?, ?, ?, ?)''',
                 (from_id, to_id, rid,
                  'test'))
testFile.close()

conn.commit()

conn.execute('''CREATE UNIQUE INDEX "main"."rid" ON "relations" ("rid" COLLATE BINARY ASC);''')
conn.execute('''CREATE UNIQUE INDEX "main"."eid" ON "entities" ("eid" COLLATE BINARY ASC);''')
conn.commit()

conn.close()
print('finished!')
