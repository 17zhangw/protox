#!/bin/bash

set -ex

rm -rf artifacts/
python3 main.py --config configs/config.yaml \
                --agent wolp \
				--model-config configs/wolp_params.yaml \
				--benchmark-config configs/benchmark/tpch.yaml \
				--seed 1580388 \
				--max-iterations 1000 \
				--duration 5 \
				--horizon 5 \
				--timeout 1 \
				--target latency \
				--reward cdb_delta
tar zcf base_0.tgz artifacts
