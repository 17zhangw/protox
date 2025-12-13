#!/bin/bash

set -ex
set -o pipefail

# Requires:
# PORT
# NAME
# ARCHIVE
# CONFIG

# Install the correct openbox dependency.
#pip uninstall -y openbox
#(cd unitune/openbox && pip install .)

NOISEPAGE_DIR=/mnt/nvme0n1/wz2/noisepage

OUTPUT_BASE=/home/wz2/mythril/unitune_runs/
mkdir -p $OUTPUT_BASE

launch_db () {
	# Remove previous pgdata.
	$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata$PORT stop || true
	rm -rf $NOISEPAGE_DIR/pgdata$PORT

	# Untar the archive
	mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata$PORT
	tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata$PORT --strip-components 1
	echo "port=$PORT" >> $NOISEPAGE_DIR/pgdata$PORT/postgresql.conf
	# Start the database.
	$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata$PORT --wait -t 300 -l $NOISEPAGE_DIR/pg.log.$PORT

	# Sleep for 5 seconds
	sleep 5
}


CONFIG="/home/wz2/mythril/scripts/experiments/dsb_workshifts/$CONFIG"

rm -rf unitune/UniTune/logs$PORT
#rm -rf /tmp/indexsize.json.$PORT
rm -rf /tmp/tmp.cnf
launch_db

(cd unitune/UniTune && python3 main.py --config-ini ${CONFIG})
mv unitune/UniTune/logs$PORT ${OUTPUT_BASE}/${NAME}.$PORT

$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata$PORT stop || true
rm -rf $NOISEPAGE_DIR/pgdata$PORT
rm -rf $NOISEPAGE_DIR/pg.log.$PORT
