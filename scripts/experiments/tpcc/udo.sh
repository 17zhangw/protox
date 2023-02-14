#!/bin/bash

set -ex
set -o pipefail

DURATION_HR=8
NOISEPAGE_DIR=/mnt/nvme0n1/wz2/noisepage
ARCHIVE=/home/wz2/mythril/data/tpcc_sf100.tgz

OUTPUT_BASE=/home/wz2/mythril/exps_tpcc/udo_runs/
mkdir -p $OUTPUT_BASE
#rm -rf $OUTPUT_BASE/*

launch_db () {
	# Remove previous pgdata.
	$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata stop || true
	rm -rf $NOISEPAGE_DIR/pgdata
	rm -rf $NOISEPAGE_DIR/pg.log

	# Untar the archive
	mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata
	tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata --strip-components 1
	# Start the database.
	$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata --wait -t 300 -l $NOISEPAGE_DIR/pg.log

	# Sleep for 5 seconds
	sleep 5
}


for ((j = 0; j < 4; j++));
do
	OUT_LOG=${OUTPUT_BASE}/tpcc_sf100_term40_$j.log
	launch_db

	cd UDO

	python3 \
		-m udo \
		-system postgres \
		-db benchbase \
		-username admin \
		-queries "/tmp/test" \
		-indices /home/wz2/mythril/UDO/params/tpcc_default/tpcc_index.txt \
		-sys_params /home/wz2/mythril/UDO/params/tpcc_default/params.json \
		--benchmark tpcc \
		--benchbase-path /home/wz2/noisepage-pilot/artifacts/benchbase \
		--benchbase-config /home/wz2/noisepage-pilot/config/behavior/benchbase/tpcc_config.xml \
		--pg-path /mnt/nvme0n1/wz2/noisepage/ \
		--pg-data "pgdata" \
		-duration 8 \
		-agent udo \
		-horizon 8 \
		-heavy_horizon 3 \
		-rl_max_delay_time 5 \
		-default_query_time_out 0 2>&1 | tee ${OUT_LOG}

	cd ..
done
