#!/bin/bash
# Experiment A matrix: 3 lanes in parallel, each lane runs its configs sequentially
cd "$(dirname "$0")"
lane() { for cfg in "$@"; do set -- $cfg; .venv/bin/python experiments/think_benchmark.py --subjects 1-10 --layout $1 --order $2 --confirm $3 > "logs/bench/${1}_${2}_${3}.log" 2>&1; done; }
lane "linear zipf none" "linear fixed none" "hier zipf none" "hier fixed none" &
lane "linear zipf repeat" "linear fixed repeat" "hier zipf repeat" "hier fixed repeat" &
lane "linear zipf imagery" "linear fixed imagery" "hier zipf imagery" "hier fixed imagery" &
wait; echo MATRIX DONE
