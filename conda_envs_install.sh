#!/bin/bash
# # Installing miniconda3
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm ~/miniconda3/miniconda.sh

# conda remove -n rp-review-env7 --all -y
source ~/miniconda3/etc/profile.d/conda.sh


# Create Conda environments
# env1
conda env create -f ./yml/rp-review-env1.yml
conda init bash
conda activate rp-review-env1
Rscript -e "devtools::install_github('IRkernel/IRkernel'); IRkernel::installspec()"

# pip install ipykernel
# python -m ipykernel install --user --name rp-review-env1 --display-name "rp-review-env1"
conda deactivate

# env2
conda env create -f ./yml/rp-review-env2.yml
conda activate rp-review-env2
pip install ipykernel
python -m ipykernel install --user --name rp-review-env2 --display-name "rp-review-env2"
conda deactivate

# env3
conda env create -f ./yml/rp-review-env3.yml
conda activate rp-review-env3
pip install ipykernel
python -m ipykernel install --user --name rp-review-env3 --display-name "rp-review-env3"
conda deactivate

# env4
conda env create -f ./yml/rp-review-env4.yml
conda activate rp-review-env4
pip install ipykernel
python -m ipykernel install --user --name rp-review-env4 --display-name "rp-review-env4"
conda deactivate


# env6
conda env create -f ./yml/rp-review-env6.yml
conda activate rp-review-env6
pip install ipykernel
python -m ipykernel install --user --name rp-review-env6 --display-name "rp-review-env6"
conda deactivate

# env7
conda env create -f ./yml/rp-review-env7.yml
conda activate rp-review-env7
pip install ipykernel
python -m ipykernel install --user --name rp-review-env7 --display-name "rp-review-env7"
conda deactivate



# env5
conda env create -f ./yml/rp-review-env5.yml
conda activate rp-review-env5
pip install ipykernel
python -m ipykernel install --user --name rp-review-env5 --display-name "rp-review-env5"

# Install bayesopt
sudo apt install -y libboost-dev cmake cmake-curses-gui g++ octave liboctave-dev freeglut3-dev || true
git clone https://github.com/rmcantin/bayesopt
cd bayesopt
cmake . && make && sudo make install
cmake -DBAYESOPT_PYTHON_INTERFACE=ON . && make && sudo make install

conda deactivate
