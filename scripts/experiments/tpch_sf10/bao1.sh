#!/bin/bash

set -ex

NOISEPAGE_DIR=/mnt/nvme0n1/wz2/noisepage

python3 scripts/experiments/tpch_sf10/load_tpch.py --config-file configs/tpch_sf10.yaml.5451
/mnt/nvme0n1/wz2/noisepage/pg_ctl stop -D /mnt/nvme0n1/wz2/noisepage/pgdata5451 || true
#rm -rf /mnt/nvme0n1/wz2/noisepage/pgdata5451
#rm -rf /mnt/nvme0n1/wz2/noisepage/pg.log.5451

#mkdir -m 0700 -p $NOISEPAGE_DIR/pgdata5451
#tar xf $ARCHIVE -C $NOISEPAGE_DIR/pgdata5451 --strip-components 1
#echo "port=5451" >> $NOISEPAGE_DIR/pgdata5451/postgresql.conf
#cp $NOISEPAGE_DIR/postgresql.auto.conf $NOISEPAGE_DIR/pgdata5451/

sed -i '$ d' /mnt/nvme0n1/wz2/noisepage/pgdata5451/postgresql.auto.conf
echo "shared_preload_libraries = 'pg_bao'" >> $NOISEPAGE_DIR/pgdata5451/postgresql.auto.conf
/mnt/nvme0n1/wz2/noisepage/pg_ctl start -D $NOISEPAGE_DIR/pgdata5451 --log=pg.5451

OUTPUT_BASE=/home/wz2/mythril/exps_tpch_sf10/bao_runs/
mkdir -p $OUTPUT_BASE
#rm -rf $OUTPUT_BASE/*

B_ARMS=( 49 )
B_DURATIONS=( 30 )
B_PQT=( 60 )

idx=0
cd Bao_1

for ((i = 0; i < ${#B_ARMS[@]}; i++));
do
	DURATION="${B_DURATIONS[i]}"
	ARMS="${B_ARMS[i]}"
	PQT="${B_PQT[i]}"

	OUT_LOG=$OUTPUT_BASE/bao_out$idx.log.5451

	cd bao_server
	python3 main.py &
	main_pid=$!
	cd ..

	sleep 15

	python3 run_queries.py \
		--qorder "/home/wz2/mythril/queries/tpch/qorder_bao.txt" \
		--duration $DURATION \
		--port 5451 \
		--num-arms $ARMS \
		--per-query-timeout $PQT | tee $OUT_LOG

	#sleep 5
	#kill -9 $main_pid
	##kill -9 $(ps aux | grep '[p]ython3 main.py' | awk '{print $2}')
	#sleep 5
	#rm -rf ~/mythril/Bao_1/bao_server/bao_previous_model
	#rm -rf ~/mythril/Bao_1/bao_server/bao_default_model
	#rm -rf ~/mythril/Bao_1/bao_server/bao.db

	idx=$((idx+1))
done

# Remove previous pgdata5451.
$NOISEPAGE_DIR/pg_ctl -D $NOISEPAGE_DIR/pgdata5451 stop || true
rm -rf $NOISEPAGE_DIR/pgdata5451
rm -rf $NOISEPAGE_DIR/pg.log.5451
