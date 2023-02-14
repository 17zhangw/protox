#!/bin/bash

set -ex

./scripts/experiments/job_full/udo.sh
./scripts/experiments/job_full/unitune.sh
./scripts/experiments/job_full/bao.sh
./scripts/experiments/job_full/bao_noindex.sh
