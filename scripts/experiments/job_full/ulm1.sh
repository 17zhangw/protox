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
	$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata5464 stop || true
	rm -rf $NOISEPAGE_DIR/pgdata5464

	# Untar the archive
	mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata5464
	tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata5464 --strip-components 1
	echo "port=5464" >> $NOISEPAGE_DIR/pgdata5464/postgresql.conf
	# Start the database.
	$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata5464 --wait -t 300 -l $NOISEPAGE_DIR/pg.log.5464

	# Sleep for 5 seconds
	sleep 5
}

CONFIGS=( "/home/wz2/mythril/scripts/experiments/job_full/ulm.1")
ARCHIVES=( "/home/wz2/mythril/data/job.tgz" )
NAMES=( "ulm1_" )

for ((j = 0; j < 1; j++));
do
	for ((i = 0; i < ${#CONFIGS[@]}; i++));
	do
		CONFIG="${CONFIGS[i]}"
		ARCHIVE="${ARCHIVES[i]}"
		NAME="${NAMES[i]}"

		rm -rf unitune/UniTune/logs
		rm -rf unitune/UniTune/logs_ulm1
		rm -rf /tmp/indexsize.json.5464
		rm -rf /tmp/tmp.cnf
		launch_db

		(cd unitune/UniTune && python3 main.py --config-ini ${CONFIG})
		mv unitune/UniTune/logs_ulm1 ${OUTPUT_BASE}/$NAME$j
	done
done

# Remove previous pgdata.
$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata5464 stop || true
rm -rf $NOISEPAGE_DIR/pgdata5464
