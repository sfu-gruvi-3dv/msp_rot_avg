# make sure the python is 3.7
conda create -n rot_avg37 python=3.7

# common libraries
conda install -c conda-forge scipy opencv scikit-image matplotlib networkx  numpy pillow tensorboard tqdm ipython

conda install -c conda-forge hdf5storage python-lmdb

# install pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# install dgl
conda install -c dglteam dgl-cuda10.2

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-$1.7.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-$1.7.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-$1.7.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-$1.7.0+cu102.html
pip install torch-geometric

conda install -c conda-forge tensorboard
pip install colorama
pip install torchgeometry
