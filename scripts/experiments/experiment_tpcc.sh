#!/bin/bash

set -ex
set -o pipefail

./scripts/experiments/tpcc/udo.sh
./scripts/experiments/tpcc/unitune.sh
