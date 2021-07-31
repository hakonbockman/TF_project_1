#!/bin/bash

SBATCH --partition=CPUQ
SBATCH --account=share-ie-idi
SBATCH --time=48:00:00
SBATCH --nodes=1
SBATCH -c 28
#SBATCH --ntasks-per-node=40

SBATCH --mail-user=haakosbo@stud.ntnu.no
SBATCH --mail-type=ALL
SBATCH --job-name="classification on UAV footage"
SBATCH --output="visual.out"

echo "SBATCH done"
echo "module loading.."
module purge
module list
module restore cuDNN804_CUDA1111_Py386_GCC1020
module list
echo "module done."

echo "Virtual env and packages loading.."
python --version
pip --version
source /cluster/work/haakosbo/env/py386_cuDNN804_CUDA1111/bin/activate
pip --version
python -m pip install --upgrade pip
pip --version
pip install tensorflow --no-cache-dir
pip install tflite-model-maker --no-cache-dir

echo "done with all packages."

echo "run code"

python main.py

echo "done"

exit
