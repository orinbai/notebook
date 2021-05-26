import psycopg2
import csv

basedir = "/home/orin/Learning/Superset"
csvname = "2021年4月竞品经销商信息列表.csv"
valueArr = ["%s"]
conn = psycopg2.connect(
    host="192.168.0.11",
    database="superset",
    user="postgres",
    password="abc123"
    )
cur = conn.cursor()
# print("PostgreSQL Version:")
# cur.execute("SELECT version()")
# print(cur.fetchone())
sql = "select column_name from information_schema.columns \
    where table_schema='public' and table_name='salesinfo'"
cur.execute(sql)
tCols = [x[0] for x in cur.fetchall()]
# cur.fetchall()
colSTR = '"' + "\", \"".join(tCols[1:]) + '"'
newVArr = valueArr*(len(tCols)-1)
print(colSTR)
# sql = "INSERT INTO salesinfo (%s) VALUES (%s)" % (colSTR, "'"+"','".join(valueArr*(len(tCols)-1))+"'")
sql = "INSERT INTO salesinfo (%s) VALUES (%s)" % (colSTR, ",".join(valueArr*(len(tCols)-1)))
lines = []
with open("%s/%s" % (basedir, csvname)) as f:
    f.readline()
    f.readline()
    m = csv.reader(f)
    value2Insert = list(m)
    value2Insert = list(map(lambda x: tuple(x), value2Insert))
try:
    print(len(newVArr), len(value2Insert[0]), len(tCols[1:]))
    print(sql, len(tCols))
    print(sql % value2Insert[0])
    # print(value2Insert[0], len(value2Insert[0]))
    cur.executemany(sql, value2Insert)
    # print(sql % value2Insert[0])
    # cur.execute(sql % value2Insert[0])
    conn.commit()
    conn.close()
    print("OK")
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()