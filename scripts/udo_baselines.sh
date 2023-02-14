#!/bin/bash

set -ex

DURATION_HR=8
NOISEPAGE_DIR=/home/wz2/noisepage-pilot/artifacts/noisepage

OUTPUT_BASE=/home/wz2/mythril/runs/
mkdir -p $OUTPUT_BASE
rm -rf $OUTPUT_BASE/*

launch_db () {
	# Remove previous pgdata.
	$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata stop || true
	rm -rf $NOISEPAGE_DIR/pgdata

	# Untar the archive
	mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata
	tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata --strip-components 1
	# Start the database.
	$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata --wait -t 300 -l $NOISEPAGE_DIR/pg.log

	# Sleep for 5 seconds
	sleep 5
}

B_QUERIES=( "/home/wz2/mythril/queries/tpch" "/home/wz2/mythril/queries/job_a" )
B_PARAMS=( "/home/wz2/mythril/UDO/params/tpch_sf1_default/params.json" "/home/wz2/mythril/UDO/params/job_default/params.json" )
B_INDEXES=( "/home/wz2/mythril/UDO/params/tpch_sf1_default/tpch_index.txt" "/home/wz2/mythril/UDO/params/job_default/job_index.txt" )
B_ARCHIVES=( "/home/wz2/mythril/data/tpch_sf1.tgz" "/home/wz2/mythril/data/job.tgz" )
B_TIMEOUTS=( 6 10)
idx=0

for ((i = 0; i < ${#B_QUERIES[@]}; i++));
do
	QUERIES="${B_QUERIES[i]}"
	PARAMS="${B_PARAMS[i]}"
	INDICES="${B_INDEXES[i]}"
	ARCHIVE="${B_ARCHIVES[i]}"
	Q_TIMEOUT="${B_TIMEOUTS[i]}"

	for ((j = 0; j < 3; j++));
	do
		OUT_LOG=${OUTPUT_BASE}/out$idx.log
		launch_db

		cd UDO
		
		python3 \
			-m udo \
			-system postgres \
			-db benchbase \
			-username admin \
			-queries $QUERIES \
			-indices $INDICES \
			-sys_params $PARAMS \
			-duration $DURATION_HR \
			-agent udo \
			-horizon 8 \
			-heavy_horizon 3 \
			-rl_max_delay_time 5 \
			-default_query_time_out $Q_TIMEOUT 2>&1 | tee ${OUT_LOG}

		cd ..

		idx=$((idx+1))
	done
done

