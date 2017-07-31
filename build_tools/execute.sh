#/usr/bin/env sh
export PATH="$HOME/miniconda/bin:$PATH"
source activate testenv

if [ ${COVERAGE} == "true" ]; then
	nosetests -c .noserc -q --cover-html-dir=build --cover-html
else
    nosetests -v
fi
