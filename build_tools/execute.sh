#/usr/bin/env sh
export PATH="$HOME/miniconda/bin:$PATH"
source activate testenv

if [ ${COVERAGE} == "true" ]; then
	mkdir build
	nosetests --with-coverage -q --cover-html-dir=build --cover-html
	coverage report -m
else
    nosetests -v -sx
fi
