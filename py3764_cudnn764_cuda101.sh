module purge
module restore py374_cuda101243_cudnn76438
module list
echo "module done."

echo "Virtual env and packages loading.."
python --version
pip --version
source /cluster/work/haakosbo/env/python374/bin/activate
pip --version
python -m pip install --upgrade pip
pip --version
pip install tensorflow==2.3.0 --no-cache-dir
pip install tflite-model-maker==0.2.5 --no-cache-dir

echo "done with all packages."

echo "run code"

python main.py