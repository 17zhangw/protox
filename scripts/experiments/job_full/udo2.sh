#!/bin/bash

set -ex
set -o pipefail

DURATION_HR=30
NOISEPAGE_DIR=/mnt/nvme0n1/wz2/noisepage

OUTPUT_BASE=/home/wz2/mythril/exps/udo_runs/
mkdir -p $OUTPUT_BASE
#rm -rf $OUTPUT_BASE/*

launch_db () {
	# Remove previous pgdata5442.
	$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata5442 stop || true
	rm -rf $NOISEPAGE_DIR/pgdata5442
	rm -rf $NOISEPAGE_DIR/pg.log.5442

	# Untar the archive
	mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata5442
	tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata5442 --strip-components 1
	echo "port=5442" >> $NOISEPAGE_DIR/pgdata5442/postgresql.conf
	# Start the database.
	$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata5442 --wait -t 300 -l $NOISEPAGE_DIR/pg.log.5442

	# Sleep for 5 seconds
	sleep 5
}

B_QUERIES=( "/home/wz2/mythril/queries/job_full" )
B_PARAMS=( "/home/wz2/mythril/UDO/params/job_default/params.json" )
B_INDEXES=( "/home/wz2/mythril/UDO/params/job_default/job_index.txt" )
B_ARCHIVES=( "/home/wz2/mythril/data/job.tgz" )
B_TIMEOUTS=( 30 )
B_HORIZONS=( 8 )
B_HHORIZONS=( 3 )
B_MAX_DELAY=( 5 )
B_NAMES=( "default_udo" )

for ((i = 0; i < ${#B_QUERIES[@]}; i++));
do
	QUERIES="${B_QUERIES[i]}"
	PARAMS="${B_PARAMS[i]}"
	INDICES="${B_INDEXES[i]}"
	ARCHIVE="${B_ARCHIVES[i]}"
	Q_TIMEOUT="${B_TIMEOUTS[i]}"
	HORIZON="${B_HORIZONS[i]}"
	HHORIZON="${B_HHORIZONS[i]}"
	MAX_DELAY="${B_MAX_DELAY[i]}"
	NAME="${B_NAMES[i]}"

	for ((j = 0; j < 4; j++));
	do
		OUT_LOG=${OUTPUT_BASE}/$NAME$j.log.5442
		launch_db

		cd UDO
		
		python3 \
			-m udo \
			-system postgres \
			-db benchbase \
			-username admin \
			-port 5442 \
			-queries $QUERIES \
			-indices $INDICES \
			-sys_params $PARAMS \
			-duration $DURATION_HR \
			-agent udo \
			-horizon $HORIZON \
			-heavy_horizon $HHORIZON \
			-rl_max_delay_time $MAX_DELAY \
			-default_query_time_out $Q_TIMEOUT 2>&1 | tee ${OUT_LOG}

		cd ..
	done
done

# Remove previous pgdata5442.
$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata5442 stop || true
rm -rf $NOISEPAGE_DIR/pgdata5442
rm -rf $NOISEPAGE_DIR/pg.log.5442
