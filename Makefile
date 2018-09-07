.PHONY: build

build:
	@scripts/build.sh

build-local:
	@scripts/build_local.sh

build-base:
	@scripts/build_base_image.sh
