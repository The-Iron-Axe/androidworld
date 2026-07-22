import sqlite3
import os

path = os.path.join(os.environ["TEMP"], "broccoli.db")
c = sqlite3.connect(path)
print("FTS options:", [r[0] for r in c.execute("pragma compile_options") if "FTS" in r[0]])
print("---")
for typ, name, sql in c.execute(
    "select type, name, sql from sqlite_master where sql is not null order by type, name"
):
  print(f"[{typ}] {name}")
  print(sql)
  print()
