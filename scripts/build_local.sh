#!/bin/bash
set -euf -o pipefail

DOCKER_IMAGE="quay.io/kairosinc/serving"
DOCKER_TAG="local"

#Check for dependencies
if ! [ -x "$(command -v docker)" ]; then
	echo "Error: couldn't find docker in this system." >&2
	exit 1
fi

#Docker login to quay
if ! grep -q "quay.io" ~/.docker/config.json ; then
	docker login "quay.io"
fi

#Build local image
docker build \
	--tag $DOCKER_IMAGE:$DOCKER_TAG \
	--compress \
	--file Dockerfile.local \
	.

# Wrap up
echo "Hooray! $DOCKER_TAG image built: $DOCKER_IMAGE:$DOCKER_TAG"
