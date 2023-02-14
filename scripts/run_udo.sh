#!/bin/bash

set -ex

DURATION_HR=5
QUERIES=/home/wzhang/mythril/queries/tpch/
PARAMS=/home/wzhang/mythril/UDO/params/tpch_sf1_default/params.json
INDICES=/home/wzhang/mythril/UDO/params/tpch_sf1_default/tpch_index.txt
ARCHIVE=/home/wzhang/mythril/data/tpch_sf1.tgz
NOISEPAGE_DIR=/home/wzhang/noisepage-pilot/artifacts/noisepage
OUT_LOG=/home/wzhang/mythril/out.log

# Remove previous pgdata.
$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata stop || true
rm -rf $NOISEPAGE_DIR/pgdata

# Untar the archive
mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata
tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata --strip-components 1
# Start the database.
$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata --wait -t 300 -l $NOISEPAGE_DIR/pg.log

# Sleep for 15 seconds
sleep 5

# Run UDO...
(cd UDO &&
python3 \
	-m udo \
	-system postgres \
	-db benchbase \
	-username admin \
	-queries ${QUERIES} \
	-indices ${INDICES} \
	-sys_params ${PARAMS} \
	-duration ${DURATION_HR} \
	-agent udo \
	-horizon 8 \
	-heavy_horizon 3 \
	-rl_max_delay_time 5 \
	-default_query_time_out 6 2>&1 | tee ${OUT_LOG}
)
