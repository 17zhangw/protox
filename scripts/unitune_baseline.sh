#!/bin/bash

set -ex

# Install the correct openbox dependency.
pip uninstall -y openbox
(cd unitune/openbox && pip install .)

NOISEPAGE_DIR=/home/wz2/noisepage-pilot/artifacts/noisepage

OUTPUT_BASE=/home/wz2/mythril/runs/
mkdir -p $OUTPUT_BASE

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

CONFIGS=( "/home/wz2/mythril/unitune/UniTune/exp_configs/tpch_sf1_knob_index_baseline.ini" "/home/wz2/mythril/unitune/UniTune/exp_configs/job_knob_index_baseline.ini" )
ARCHIVES=( "/home/wz2/mythril/data/tpch_sf1.tgz" "/home/wz2/mythril/data/job.tgz" )

for ((i = 0; i < ${#CONFIGS[@]}; i++));
do
	CONFIG="${CONFIGS[i]}"
	ARCHIVE="${ARCHIVES[i]}"
	for ((j = 0; j < 3; j++));
	do
		rm -rf unitune/UniTune/logs
		rm -rf /tmp/indexsize.json
		rm -rf /tmp/tmp.cnf
		launch_db

		(cd unitune/UniTune && python3 main.py --config-ini ${CONFIG})
		mv unitune/UniTune/logs ${OUTPUT_BASE}/log$idx
		idx=$((idx+1))
	done
done
