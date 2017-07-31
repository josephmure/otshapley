#/usr/bin/env sh
export PATH="$HOME/miniconda/bin:$PATH"
source activate testenv

nosetests -v