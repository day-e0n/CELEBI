# nsys 대상용 자체 완결 스크립트. 환경변수를 안에서 설정하므로 nsys 쪽에 아무것도 안 넣어도 된다.
import os, time
ROOT = "/home/dy1013/sirius_multi-gpu"
# 확장 로드 시점에 읽히므로 import 전에 설정해야 한다.
os.environ["SIRIUS_CONFIG_FILE"] = f"{ROOT}/test/cpp/integration/integration.yaml"
os.environ.setdefault("SIRIUS_LOG_DIR", "/tmp/sirius_log")   # repo의 log/ 는 root 소유라 피한다
os.makedirs("/tmp/sirius_log", exist_ok=True)

import duckdb
con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
con.execute(f"LOAD '{ROOT}/build/release/extension/sirius/sirius.duckdb_extension'")
con.execute("""CREATE VIEW lineitem AS SELECT * FROM
  read_parquet('/mnt/nvme/sirius_tpch/jcch_parquet_sf50_optimized/lineitem.parquet')""")
con.execute("SET gpu_execution = true")

SQL = "SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY l_returnflag"

# 3초짜리 한 번이면 nsys가 붙기 전에 끝나 "exited before profiling started"가 난다.
# 반복해서 30초 이상 돌리고, 첫 회는 콜드이므로 이후 회차가 정상 상태 표본이 된다.
N = int(os.environ.get("Q_ITERS", "12"))
for i in range(N):
    t0 = time.time()
    rows = con.execute(SQL).fetchall()
    print(f"iter {i}: {time.time()-t0:.3f}s rows={len(rows)}", flush=True)
print("done", flush=True)
