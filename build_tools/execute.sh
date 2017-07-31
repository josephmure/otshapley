#/usr/bin/env sh
export PATH="$HOME/miniconda/bin:$PATH"
source activate testenv

if [ ${COVERAGE} == "true" ]; then
	nosetests -v --with-coverage
else
    nosetests -v
fi
