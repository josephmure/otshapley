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
conda create -n testenv --quiet --yes python=$PYTHON_VERSION nose pip numpy pandas matplotlib seaborn scikit-learn tensorflow gpy git

source activate testenv

pushd .
cd
git clone https://github.com/GPflow/GPflow.git
cd GPflow
pip install -e.
popd

conda install --quiet --yes -c conda-forge openturns
pip install -q -r requirements.txt

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls pytest pytest-cov
fi

pushd .
cd
git clone https://github.com/scikit-optimize/scikit-optimize
pip install scikit-optimize
popd

python setup.py install
