#!/bin/bash
set -euf -o pipefail

DOCKER_IMAGE="quay.io/kairosinc/serving"
TIMESTAMP=$(date +%s)

#Check for dependencies
if ! [ -x "$(command -v docker)" ]; then
	echo "Error: couldn't find docker in this system." >&2
	exit 1
fi

#Docker login to quay
if ! grep -q "quay.io" ~/.docker/config.json ; then
	docker login "quay.io"
fi

#Build base images
docker build \
	--tag $DOCKER_IMAGE:base-cpu-$TIMESTAMP\
	--compress \
	--file Dockerfile.base-cpu \
	.
docker build \
	--tag $DOCKER_IMAGE:base-$TIMESTAMP \
	--compress \
	--file Dockerfile.base \
	.

docker tag \
	$DOCKER_IMAGE:base-cpu-$TIMESTAMP\
	$DOCKER_IMAGE:base-cpu-latest
docker tag \
	$DOCKER_IMAGE:base-$TIMESTAMP\
	$DOCKER_IMAGE:base-latest

#Build local image
docker push $DOCKER_IMAGE:base-cpu-$TIMESTAMP
docker push $DOCKER_IMAGE:base-$TIMESTAMP
docker push $DOCKER_IMAGE:base-latest
docker push $DOCKER_IMAGE:base-cpu-latest

# Wrap up
echo "Hooray! Base images built:"
echo "	$DOCKER_IMAGE:base-cpu-$TIMESTAMP"
echo "	$DOCKER_IMAGE:base-cpu-latest"
echo "	$DOCKER_IMAGE:base-$TIMESTAMP"
echo "	$DOCKER_IMAGE:base-latest"
