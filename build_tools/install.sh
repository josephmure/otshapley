#/usr/bin/env sh
# Install conda using miniconda
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]; then
	if [[ "$PYTHON_VERSION" == "2.7" ]]; then
		wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
	else
		wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
	fi
fi
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda update --quiet --yes conda
popd

# Create a conda env and install packages
conda create -n testenv --quiet --yes python=$PYTHON_VERSION nose pip matplotlib pandas seaborn

source activate testenv

pip install -q scikit-optimize
conda install --quiet --yes -c conda-forge openturns

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

python setup.py install

python -c "import matplotlib.pyplot as plt"
