#!/bin/bash

set -ex
set -o pipefail

# Requires:
# BENCHMARK
# CONFIG
# BENCHMARK_CONFIG
# HYRISE_LOG
# OUTPUT
# PORT
# BAO_PORT
# PQT
# DURATION
# QORDER

NOISEPAGE_DIR=/mnt/nvme0n1/wz2/noisepage

python3 scripts/hyrise_load.py \
	--benchmark $BENCHMARK \
	--config-file $CONFIG \
	--benchmark-config-file $BENCHMARK_CONFIG \
	--hyrise-log $HYRISE_LOG

/mnt/nvme0n1/wz2/noisepage/pg_ctl stop -D /mnt/nvme0n1/wz2/noisepage/pgdata$PORT || true
sed -i '$ d' /mnt/nvme0n1/wz2/noisepage/pgdata$PORT/postgresql.auto.conf
echo "shared_preload_libraries = 'pg_bao'" >> $NOISEPAGE_DIR/pgdata$PORT/postgresql.auto.conf
/mnt/nvme0n1/wz2/noisepage/pg_ctl start -D $NOISEPAGE_DIR/pgdata$PORT

cd Bao
OUT_LOG=$OUTPUT/$PORT.log

cd bao_server
# Attempt to start Bao.
python3 main.py --port $BAO_PORT &
sleep 15

# Purge the archives.
python3 baoctl.py --clear --bao-db "${BAO_PORT}.db" --port $BAO_PORT
cd ..

python3 run_queries.py \
	--qorder $QORDER \
	--duration $DURATION \
	--port $PORT \
	--bao-port $BAO_PORT \
	--bao-db "${BAO_PORT}.db" \
	--num-arms 49 \
	--per-query-timeout $PQT 2>&1 | tee $OUT_LOG

# Remove previous pgdata5450.
$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata$PORT stop || true
rm -rf $NOISEPAGE_DIR/pgdata$PORT
rm -rf $NOISEPAGE_DIR/pg.log.$PORT
