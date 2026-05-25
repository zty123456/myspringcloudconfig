#!/usr/bin/env bash
set -e

export PYTHONPATH=python
PYTHON=.venv/bin/python

echo "Start first training_search_util.py..."
$PYTHON python/zrt/training/search/training_search_util.py

echo "First finished. Sleep 5 minutes..."
sleep 300

echo "Start second training_search_util_1.py..."
$PYTHON python/zrt/training/search/training_search_util_1.py

echo "Both finished."