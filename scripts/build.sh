#!/bin/bash
set -euf -o pipefail

DOCKER_IMAGE="quay.io/kairosinc/serving"
BRANCH=${GIT_BRANCH#*/}

case ${BRANCH} in
	master)
		ENV=stage
		;;
	*)
		ENV=dev
		;;
esac

if [[ ${GIT_BRANCH#*}/ == "/" && ${GIT_COMMIT:0:7} == "" ]]; then
	TAG=latest
else
	TAG=${GIT_BRANCH#*/}-${GIT_COMMIT:0:7}${BUILD_NUMBER}
fi

#Check for dependencies
if ! [ -x "$(command -v docker)" ]; then
	echo "Error: couldn't find docker in this system." >&2
	exit 1
fi

#Docker login to quay
docker login -u="${QUAY_USERNAME}" -p="${QUAY_PASSWORD}" quay.io || true

#Build images
docker build \
	--pull \
	--tag $DOCKER_IMAGE:$TAG\
	--compress \
	--file Dockerfile.gpu \
	.

docker build \
	--tag $DOCKER_IMAGE:cpu-$TAG\
	--compress \
	--file Dockerfile.cpu\
	.

#Tag and push images
docker push $DOCKER_IMAGE:$TAG
docker push $DOCKER_IMAGE:cpu-$TAG

if [[ ${GIT_BRANCH#*/} == "master" ]]; then
	docker tag $DOCKER_IMAGE:cpu-$TAG $DOCKER_IMAGE:cpu-latest
	docker tag $DOCKER_IMAGE:$TAG $DOCKER_IMAGE:latest
	docker push $DOCKER_IMAGE:latest
	docker push $DOCKER_IMAGE:cpu-latest
fi

#Cleaning up
# docker rmi $DOCKER_IMAGE:latest
# docker rmi $DOCKER_IMAGE:cpu-latest
# docker rmi $DOCKER_IMAGE:$TAG
# docker rmi $DOCKER_IMAGE:cpu-$TAG
