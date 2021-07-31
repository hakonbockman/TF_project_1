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