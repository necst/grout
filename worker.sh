#!/bin/bash
# shellcheck disable=SC2164

export JAVA_HOME="$GRAAL_HOME"
export PATH=$JAVA_HOME/bin:$PATH

if [ -z "$1" ]
    then
        PORT=1600
    else
        PORT=$1
fi

cd ./worker
mvn clean
mvn package