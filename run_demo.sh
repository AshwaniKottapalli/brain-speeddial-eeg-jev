#!/bin/bash
# One command: replays public-dataset EEG (Shin 2017, subject 3) as if live, decodes "think ORDER FOOD", confirms, mock agent acts.
cd "$(dirname "$0")"
[ -d .venv ] || { uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt; }
exec .venv/bin/python demo.py "$@"
