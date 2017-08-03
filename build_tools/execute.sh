#/usr/bin/env sh
export PATH="$HOME/miniconda/bin:$PATH"
source activate testenv

if [ ${COVERAGE} == "true" ]; then
	pytest --cov-report html:build --cov=shapley
	coverage report -m
else
    nosetests -v -sx
fi
