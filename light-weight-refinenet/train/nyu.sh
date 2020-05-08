#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python3.6 src/train.py \
    --enc 152

