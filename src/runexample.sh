#!/usr/bin/env bash

set -e

# Run from this directory
cd ${0%/*}

source ../versions.sh

eval "$(pyenv init -)"
pyenv shell "$pyver"-debug

python setup.py build_ext --inplace
cygdb . -- --args python main.py

# Note:
# $ eval "$(pyenv init -)"
# $ pyenv shell "$pyver"-debug
# $ python main.py
# will execute main.py
#
# Note: Inserting the line "breakpoint()" in main.py allows stepping through the code of main.py, but does not allow stepping through the code of mylib.pyx
