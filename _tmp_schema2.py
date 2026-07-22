import subprocess
import sqlite3
import tempfile
import os
import shutil

adb = r"D:\Data\Android\platform-tools\adb.exe"
remote = "/data/data/com.flauschcode.broccoli/databases/broccoli"
tmpdir = tempfile.mkdtemp()
local = os.path.join(tmpdir, "broccoli")

# checkpoint wal into main db on device if possible
subprocess.run(
    [adb, "shell", f"sqlite3 {remote} 'PRAGMA wal_checkpoint(FULL);'"],
    check=False,
)
for suffix in ("", "-wal", "-shm"):
  src = remote + suffix
  dst = local + suffix
  subprocess.run([adb, "pull", src, dst], check=False)

print("sizes:", {f: os.path.getsize(os.path.join(tmpdir, f)) for f in os.listdir(tmpdir)})
conn = sqlite3.connect(local)
print("FTS:", [r[0] for r in conn.execute("pragma compile_options") if "FTS" in r[0]])
for row in conn.execute(
    "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
):
  print("=" * 60)
  print(row[0], row[1])
  print(row[2])
conn.close()
shutil.rmtree(tmpdir, ignore_errors=True)
