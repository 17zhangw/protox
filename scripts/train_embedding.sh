#!/bin/bash

set -ex

rm -rf artifacts/
mkdir artifacts/
cp -r out.parquet artifacts/

python3 index_embedding.py train \
		--compile \
        --output-dir artifacts \
		--config embeddings/config.json \
        --run-initial \
        --num-trials 30 \
        --benchmark-config configs/benchmark/tpch.yaml \
        --train-size 0.8 \
        --iterations-per-epoch 8000 \

		#--ray \
		#--max-concurrent 12 \
		#--mythril-dir "/home/wz2/mythril" \
		#--num-threads 20

python3 index_embedding.py eval \
        --benchmark-config configs/benchmark/tpch.yaml \
        --models artifacts \
        --intermediate-step 1 \
        --dataset artifacts/out.parquet \
        --batch-size 8192 \
        --num-batches 10

python3 index_embedding.py eval \
        --benchmark-config configs/benchmark/tpch.yaml \
        --models artifacts \
        --intermediate-step 1 \
        --dataset artifacts/out.parquet \
        --batch-size 8192 \
        --num-batches 10 \
		--eval-reconstruction \
		--index-length 0 \
		--index-sample 8192 \
		--recon-batch 8192 \
		--recon-output recon

python3 index_embedding.py plot \
        --input-dirs "artifacts" \
        --num-neighbors 3 \
        --num-data-points 8192 \
        --output-name "cluster8192_3"

python3 index_embedding.py plot \
        --input-dirs "artifacts" \
        --num-neighbors 5 \
        --num-data-points 8192 \
        --output-name "cluster8192_5"

python3 index_embedding.py plot \
        --input-dirs "artifacts" \
        --num-neighbors 10 \
        --num-data-points 8192 \
        --output-name "cluster8192_10"

python3 index_embedding.py plot \
        --input-dirs "artifacts" \
        --num-neighbors 3 \
        --num-data-points 24576 \
        --output-name "cluster24576_3"

python3 index_embedding.py plot \
        --input-dirs "artifacts" \
        --num-neighbors 5 \
        --num-data-points 24576 \
        --output-name "cluster24576_5"

python3 index_embedding.py plot \
        --input-dirs "artifacts" \
        --num-neighbors 10 \
        --num-data-points 24576 \
        --output-name "cluster24576_10"
