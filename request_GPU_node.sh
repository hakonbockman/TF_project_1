#!/bin/bash
salloc --partition=GPUQ --account=share-ie-idi --time=24:00:00 --gres=gpu:P100:2 --nodes=1 --ntasks-per-node=30 --mail-user=haakosbo@stud.ntnu.no --mail-type=ALL --job-name="GPU-RUN"