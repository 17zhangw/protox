#!/bin/bash

set -ex

## Install the correct openbox dependency.
#pip uninstall -y openbox
#(cd unitune/openbox && pip install .)

NOISEPAGE_DIR=/mnt/nvme0n1/wz2/noisepage

OUTPUT_BASE=/home/wz2/mythril/exps/unitune_runs/
mkdir -p $OUTPUT_BASE
#rm -rf $OUTPUT_BASE/*

launch_db () {
	# Remove previous pgdata.
	$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata5433 stop || true
	rm -rf $NOISEPAGE_DIR/pgdata5433

	# Untar the archive
	mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata5433
	tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata5433 --strip-components 1
	echo "port=5433" >> $NOISEPAGE_DIR/pgdata5433/postgresql.conf
	# Start the database.
	$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata5433 --wait -t 300 -l $NOISEPAGE_DIR/pg.log.5433

	# Sleep for 5 seconds
	sleep 5
}

CONFIGS=( "/home/wz2/mythril/scripts/experiments/job_full/job.2")
ARCHIVES=( "/home/wz2/mythril/data/job.tgz" )
NAMES=( "unitune_highmem0_" )

for ((j = 0; j < 2; j++));
do
	for ((i = 0; i < ${#CONFIGS[@]}; i++));
	do
		CONFIG="${CONFIGS[i]}"
		ARCHIVE="${ARCHIVES[i]}"
		NAME="${NAMES[i]}"

		rm -rf unitune/UniTune/logs
		rm -rf unitune/UniTune/logs_highmem0
		rm -rf /tmp/indexsize.json.5433
		rm -rf /tmp/tmp.cnf
		launch_db

		(cd unitune/UniTune && python3 main.py --config-ini ${CONFIG})
		mv unitune/UniTune/logs_highmem0 ${OUTPUT_BASE}/$NAME$j
	done
done
