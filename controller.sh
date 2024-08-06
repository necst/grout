#!/bin/bash

export JAVA_HOME="$LABSJDK_HOME"
export PATH=$JAVA_HOME/bin:$PATH

cd "$GRCUDA_DIST_HOME"/controller || exit;

# CLEAN and BUILD the project
mx clean;
mx build;

# Make space for the .jar to be added in GraalVM's languages
mkdir -p "$GRAAL_HOME"/languages/controller;
rm "$GRAAL_HOME"/languages/controller/controller.jar;
cp "$GRCUDA_DIST_HOME"/controller/mxbuild/dists/controller.jar "$GRAAL_HOME"/languages/controller/.;


cd "$GRCUDA_DIST_HOME"/controller || exit;
