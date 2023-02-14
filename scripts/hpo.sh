#!/bin/bash

set -ex

python3 hpo.py  --config /home/wz2/mythril/configs/config.yaml \
	--agent wolp \
	--model-config /home/wz2/mythril/configs/wolp_params.yaml \
	--benchmark-config /home/wz2/mythril/configs/benchmark/tpch.yaml \
	--mythril-dir "/home/wz2/mythril" \
	--num-trials 4 \
	--max-concurrent 4 \
	--max-iterations 1000 \
	--horizon 5 \
	--duration 30.0 \
	--target latency \
	--data-snapshot-path "data/tpch_sf10.tgz" \
	--workload-timeout 600 \
	--timeout 30 \
	--benchbase-config-path "/home/wz2/noisepage-pilot/config/behavior/benchbase/tpch_config.xml" \
	--initial-configs params.json \
	--initial-repeats 4 \

