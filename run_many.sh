#!/bin/bash

NUM_RUNS=200
for i in $(seq 1 $NUM_RUNS)
do
   echo "Run #$i"
   python run.py --seed=$i
done
