# Release 1.11.0-rc0

## Major Features and Improvements

* Prometheus exporter for TF metrics (see https://github.com/tensorflow/serving/commit/021efbd3281aa815cab0b35eab6d6d25249c12d4 for details).
* Added new REST API to [get status of model(s)](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md#model-status-api) from ModelServer.

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Built against TensorFlow [1.11.0-rc0](https://github.com/tensorflow/tensorflow/releases/tag/v1.11.0-rc0).
* Directly import tensor.proto.h (the transitive import will be removed from tensor.h soon)
* Building optimized TensorFlow Serving binaries is now easier (see [docs](https://github.com/tensorflow/serving/g3doc/setup.md]) for details)
* Adds columnar format support for input/output tensors in Predict REST API (fixes #1047)
* Development Dockerfiles now produce a more optimized ModelServer
* Fixed TensorFlow Serving API PyPi package overwriting TensorFlow package.

# Release 1.10.0

## Major Features and Improvements

* No major features or improvements.

## Breaking Changes

* TensorFlow Serving API now uses gRPC's GA release. The beta gRPC API has been deprecated, and will be removed in a future version of TensorFlow Serving. Please update your gRPC client code ([sample](https://github.com/tensorflow/serving/commit/aa35cfdb24016f6d88f82c53d45c8ce9fa550499#diff-e7d756a12c65a8b5ac90229b23523023))
* Docker images for GPU are built against NCCL 2.2, in following with Tensorflow 1.10.

## Bug Fixes and Other Changes

* Built against TensorFlow 1.10.
* Added GPU serving Docker image.
* Repo cloning and shell prompt in example readme.
* Updated Docker instructions.
* Updated min Bazel version (0.15.0).
* Convert TF_CHECK_OKs to TF_ASSERT_OK in some unit tests.
* Remove error suppression (.IgnoreError()) from BasicManager.
* Add new bazel_in_docker.sh tool for doing hermetic bazel builds.
* Fix erroneous formatting of numbers in REST API output that are larger than 6 digits.
* Add support for Python 3 while also compatible with Python 2.7 in mnist_saved_model.py.
* Fix an incorrect link to Dockerfile.devel-gpu.
* Add util for get model status.
* Adding support for secure channel to ModelServer.
* Add version output to model server binary.
* Change ServerRequestLogger::Update to only create new and delete old loggers if needed.
* Have the Model Server interpret specific hard-coded model version labels "stable" and "canary" as the smallest and largest version#, respectively.
* Add half_plus_two CPU and GPU models to test data.

# Release 0.4.0

Initial release of TensorFlow Serving.
