# Release 1.12.0

## Major Features and Improvements
* Add new REST API to [get model status](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md#model-status-api) from ModelServer (commit: 00e459f1604c40c073cbb9cb92d72cb6a88be9cd)
* Add new REST API to [get model metadata](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/api_rest.md#model-metadata-api) from ModelServer (fixes #1115) (commit: 97687024c3b7515d2f2979c35054f44c8f84d146)
* Support accepting gzipped REST API requests (fixes #1091) (commit: b94f6c89335782a7f175e8973c4f326375c55120)

## Breaking Changes

None

## Bug Fixes and Other Changes

* Update MKL build (commit: e11bd51540212242911dae00c8507e2852a5ad5a)
* Remove version pinning on pip packages (commit: 462072c2d78124c2769f820f7b63ee086de4e305)
* Update basic serving tutorials (commit: 33a4b052cedc39c21107bc99a090b59ca64ec568)
* Replacing legacy_init_op argument in SavedModelBuilder with main_op. (commit: 2fda31f905eefd2d108e9c84b8d7d55e4e482833)
* Add git hash for version metadata of model server and add tags for dev and nightly builds. (commit: 5c7740fc3d8d5c017643a8cc40a7202717b10dd6)
* Add error messages for specific cases when json for REST requests (commit: a17c89202e68bf19f369b9cbc97db7ced283b874)
* Python examples now run in a hermetic environment with all required dependencies (commit: 793fd90ee41ac34fa4c9261eef2d2c908dca9735)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Charles Verge, demfier, Kamidi Preetham, Lihang Li, naurril, vfdev, Yu Zheng

# Release 1.11.1

## Bug Fixes and Other Changes

* Fix version of model server binary (Fixes #1134)
* Range check floating point numbers correctly (Fixes #1136).
* Fix docker run script for same user and group name (Fixes #1137).
* Fix GPU build (Fixes #1150)

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

vfdev

# Release 1.11.0

## Major Features and Improvements

* Prometheus exporter for TF metrics (see https://github.com/tensorflow/serving/commit/021efbd3281aa815cab0b35eab6d6d25249c12d4 for details).

## Breaking Changes

* No breaking changes

## Bug Fixes and Other Changes

* Built against TensorFlow [1.11.0](https://github.com/tensorflow/tensorflow/releases/tag/v1.11.0)
* Accept integers for float/doubles in JSON REST API requests
* TF Serving API is now pre-built into Docker development images
* GPU Docker images are now built against cuDNN 7.2
* Add `--max_num_load_retries` flag to ModelServer (fixes #1099)
* Add user-configured model version labels to the stand-alone ModelServer binary.
* Directly import tensor.proto.h (the transitive import will be removed from tensor.h soon)
* Building optimized TensorFlow Serving binaries is now easier (see [docs](https://github.com/tensorflow/serving/g3doc/setup.md]) for details)
* Adds columnar format support for input/output tensors in Predict REST API (fixes #1047)
* Development Dockerfiles now produce a more optimized ModelServer
* Fixed TensorFlow Serving API PyPi package overwriting TensorFlow package.

## Thanks to our Contributors

This release contains contributions from many people at Google, as well as:

Feisan, joshua.horowitz, Prashanth Reddy Basani, tianyapiaozi, Vamsi Sripathi, Yu Zheng

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
