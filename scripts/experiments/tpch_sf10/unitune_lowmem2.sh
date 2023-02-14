#!/bin/bash

set -ex

# Install the correct openbox dependency.
#pip uninstall -y openbox
#(cd unitune/openbox && pip install .)

NOISEPAGE_DIR=/mnt/nvme0n1/wz2/noisepage

OUTPUT_BASE=/home/wz2/mythril/exps_tpch_sf10/unitune_runs/
mkdir -p $OUTPUT_BASE
#rm -rf $OUTPUT_BASE/*

launch_db () {
	# Remove previous pgdata.
	$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata5434 stop || true
	rm -rf $NOISEPAGE_DIR/pgdata5434

	# Untar the archive
	mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata5434
	tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata5434 --strip-components 1
	echo "port=5434" >> $NOISEPAGE_DIR/pgdata5434/postgresql.conf
	# Start the database.
	$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata5434 --wait -t 300 -l $NOISEPAGE_DIR/pg.log.5434

	# Sleep for 5 seconds
	sleep 5
}

CONFIGS=( "/home/wz2/mythril/scripts/experiments/tpch_sf10/tpch.2" )
ARCHIVES=( "/home/wz2/mythril/data/tpch_sf10.tgz" )
NAMES=( "unitune_lowmem2_" )

for ((j = 0; j < 1; j++));
do
	for ((i = 0; i < ${#CONFIGS[@]}; i++));
	do
		CONFIG="${CONFIGS[i]}"
		ARCHIVE="${ARCHIVES[i]}"
		NAME="${NAMES[i]}"
		rm -rf unitune/UniTune/logs
		rm -rf unitune/UniTune/logs_lowmem2
		rm -rf /tmp/indexsize.json.5434
		rm -rf /tmp/tmp.cnf
		launch_db

		(cd unitune/UniTune && python3 main.py --config-ini ${CONFIG})
		mv unitune/UniTune/logs_lowmem2 ${OUTPUT_BASE}/$NAME$j
	done
done
