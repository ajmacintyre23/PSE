#!/bin/bash
#SBATCH --job-name=install
#SBATCH --account=zeelab
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=20gb

# Load necessary Hyak modules
module load intel/oneAPI/2021.1.1
module load ssmc/miniconda/3.9
module load cuda/11.8.0
module load cmake/3.20.0
module load gcc/10.2.0

# Load hoomd by activating the python virtual environment in which HOOMD was built.
# Different file path needed for different users.
source /gscratch/zeelab/ajmac/installations/polydisperse/hoomd/hoomd-venv/bin/activate
rm -r build
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=/gscratch/zeelab/ajmac/installations/polydisperse/hoomd/hoomd-venv/lib/python3.9/site-packages/hoomd \
        -DCOPY_HEADERS=ON \
        -DCMAKE_CXX_FLAGS=-march=native \
        -DCMAKE_C_FLAGS=-march=native \
        -DENABLE_CUDA=ON
make install
