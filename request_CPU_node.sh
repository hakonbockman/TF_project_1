#!/bin/bash
salloc --partition=CPUQ --account=share-ie-idi --time=48:00:00 --nodes=1 --ntasks-per-node=45 --mail-user=haakosbo@stud.ntnu.no --mail-type=ALL --job-name="CPU-RUN-40Threads"