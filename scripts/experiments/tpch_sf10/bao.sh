#!/bin/bash

set -ex


python3 scripts/experiments/tpch_sf10/load_tpch.py
/mnt/nvme0n1/wz2/noisepage/pg_ctl stop -D /mnt/nvme0n1/wz2/noisepage/pgdata

cd Bao/pg_extension
PATH="/home/wz2/noisepage-pilot/build/noisepage/build/bin:$PATH" make USE_PGXS=1 clean
PATH="/home/wz2/noisepage-pilot/build/noisepage/build/bin:$PATH" make USE_PGXS=1 install
sed -i '$ d' /mnt/nvme0n1/wz2/noisepage/pgdata/postgresql.auto.conf
echo "shared_preload_libraries = 'pg_bao'" >> /mnt/nvme0n1/wz2/noisepage/pgdata/postgresql.auto.conf
/mnt/nvme0n1/wz2/noisepage/pg_ctl start -D /mnt/nvme0n1/wz2/noisepage/pgdata
cd ../../

OUTPUT_BASE=/home/wz2/mythril/exps_tpch_sf10/bao_runs/
mkdir -p $OUTPUT_BASE
rm -rf $OUTPUT_BASE/*

B_ARMS=( 49 49 49 49 )
B_DURATIONS=( 8 8 8 8 )
B_PQT=( 60 60 60 60 )

idx=0
cd Bao

for ((i = 0; i < ${#B_ARMS[@]}; i++));
do
	DURATION="${B_DURATIONS[i]}"
	ARMS="${B_ARMS[i]}"
	PQT="${B_PQT[i]}"

	OUT_LOG=$OUTPUT_BASE/bao_out$idx.log

	cd bao_server
	python3 main.py &
	cd ..

	python3 run_queries.py \
		--qorder "/home/wz2/mythril/queries/tpch/qorder_bao.txt" \
		--duration $DURATION \
		--port 5432 \
		--num-arms $ARMS \
		--per-query-timeout $PQT | tee $OUT_LOG

	sleep 5
	kill -9 $(ps aux | grep '[p]ython3 main.py' | awk '{print $2}')
	sleep 5
	rm -rf ~/mythril/Bao/bao_server/bao_previous_model
	rm -rf ~/mythril/Bao/bao_server/bao_default_model
	rm -rf ~/mythril/Bao/bao_server/bao.db

	idx=$((idx+1))
done
