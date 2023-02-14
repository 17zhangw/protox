#!/bin/bash

set -ex

# Install the correct openbox dependency.
pip uninstall -y openbox
(cd unitune/openbox && pip install .)

NOISEPAGE_DIR=/home/wzhang/noisepage-pilot/artifacts/noisepage

# Remove previous pgdata.
$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata stop || true
rm -rf $NOISEPAGE_DIR/pgdata

rm -rf /tmp/indexsize.json
rm -rf unitune/UniTune/logs
mkdir -p unitune/UniTune/logs/results

# Untar the archive
mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata
tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata --strip-components 1
# Start the database.
$NOISEPAGE_DIR/pg_ctl start -D $NOISEPAGE_DIR/pgdata --wait -t 300 -l $NOISEPAGE_DIR/pg.log

# Sleep for 15 seconds
sleep 5

# Run UDO...
(cd unitune/UniTune && python3 main.py)
