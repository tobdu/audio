#!/bin/bash

usage() {
  cat <<-EOF
    Managing bash dependencies

    assert:  if not installed -> fail
    install: if not installed -> install

    Usage:
    ./dependencies.sh <assert|install> <dep1> <dep2> ...

EOF
}

assert(){
    if [[ $(command -v $1) == "" ]]
    then
		echo "please install $1"
		exit 1
	else
		echo "$1 found"
	fi
}

install(){
    if [[ $(command -v $1) == "" ]]
    then
        echo "installing $1"
        brew install $1
    else
        echo "$1 found"
    fi
}

CMD=$1
shift

if [[ ! "${CMD}" = "assert" && ! "${CMD}" = "install" ]]; then
    usage
    exit 1
fi


for exe in $@
do
    ${CMD} $exe
done
